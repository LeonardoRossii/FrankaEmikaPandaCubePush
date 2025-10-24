import numpy as np
import scipy.sparse as sp
import osqp

class CollisionQPFilter:
    def __init__(self, env, cbf_modules):
        self.env = env
        self.modules = cbf_modules

    @staticmethod
    def _solve_qp(u_des, a_list, b_list, P_extra=None, q_extra=None,
                  *, drop_tol=1e-10, normalize_rows=True):
        u_des = np.asarray(u_des, dtype=float).reshape(-1)
        n = u_des.size

        A_clean, b_clean = [], []
        for a, b in zip(a_list, b_list):
            a = np.asarray(a, dtype=float).reshape(n)
            b = float(b)
            if not np.all(np.isfinite(a)) or not np.isfinite(b):
                continue
            norm = np.linalg.norm(a)
            if norm < drop_tol:
                continue
            if normalize_rows and norm > 0:
                a = a / norm
                b = b / norm
            A_clean.append(a)
            b_clean.append(b)

        m = len(A_clean)
        P = sp.eye(n, format="csc")
        q = -u_des.copy()

        if P_extra is not None and q_extra is not None:
            P = (P + sp.csc_matrix(P_extra)) if not isinstance(P_extra, sp.csc_matrix) else (P + P_extra)
            q = (q + q_extra)

        if m == 0:
            if P_extra is None:
                u = u_des
            else:
                M = (sp.eye(n, format="csc") + sp.csc_matrix(P_extra)).tocsc()
                rhs = u_des - q_extra
                u = np.linalg.solve(M.toarray(), rhs)
            return u, np.zeros(0, dtype=float), True

        A_mat = np.stack(A_clean, axis=0)
        b_arr = np.array(b_clean, dtype=float)

        Au = sp.csc_matrix(A_mat)
        l = b_arr
        uvec = np.full(m, np.inf, dtype=float)

        prob = osqp.OSQP()
        prob.setup(P=P, q=q, A=Au, l=l, u=uvec,
                   verbose=False, polish=False, eps_abs=1e-6, eps_rel=1e-6)
        res = prob.solve()

        ok = (res.info.status_val in (1, 2)) and (res.x is not None)
        if not ok:
            return u_des, np.zeros(m, dtype=float), False

        x = np.array(res.x, dtype=float)
        y = np.array(res.y, dtype=float) if res.y is not None else np.zeros(m, dtype=float)
        return x, y, True

    def apply(self, u_des, env):
        self.env = env
        u_act = np.array(u_des, dtype=float).copy()
        u_nom = u_act[:7]

        all_As, all_Bs = [], []
        groups = []
        row_start = 0

        for mod in self.modules:
            a_list, b_list = mod.constraints(self.env)
            k = len(a_list)
            if k:
                all_As.extend(a_list)
                all_Bs.extend(b_list)
                groups.append((mod.__class__.__name__, row_start, row_start + k))
                row_start += k

        P_extra_total = None
        q_extra_total = None
        for mod in self.modules:
            obj = getattr(mod, "objective_terms", None)
            if obj is None:
                continue
            out = mod.objective_terms(self.env)
            if out is None:
                continue
            P_e, q_e = out
            P_extra_total = (P_e if P_extra_total is None else P_extra_total + P_e)
            q_extra_total = (q_e if q_extra_total is None else q_extra_total + q_e)

        u_safe, y, ok = self._solve_qp(u_nom, all_As, all_Bs,
                                       P_extra=P_extra_total, q_extra=q_extra_total)

        u_act[:7] = u_safe if ok else u_nom

        efforts = {}
        if len(all_As) > 0:
            A_mat = np.stack([np.asarray(ai, dtype=float).reshape(-1) for ai in all_As], axis=0)
            for name, lo, hi in groups:
                A_g = A_mat[lo:hi, :]
                y_g = y[lo:hi] if y.size >= hi else np.zeros(hi - lo, dtype=float)
                delta_g = -A_g.T @ y_g
                efforts[name] = {
                    "delta": delta_g,
                    "effort_l2": float(np.linalg.norm(delta_g)),
                    "active_constraints": int(np.count_nonzero(y_g)),
                }
        return u_act, efforts
