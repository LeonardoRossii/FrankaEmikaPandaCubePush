import numpy as np
import scipy.sparse as sp
import osqp

class QPFilter:
    """
    Quadratic Programming (QP) based on the CBF safety filter modules
    Given a desired control input u_des from the policy, it enforces 
    all the constraints from the provided modules.
    If the QP is infeasible or fails, the filter returns the original u_des.
    """
    def __init__(self, env, cbf_modules):
        self.env = env
        self.modules = cbf_modules

    @staticmethod
    def _solve_qp(u_des, a_list, b_list, P_extra=None, q_extra=None,
                  *, drop_tol=1e-10, normalize_rows=True):

        # Ensure u_des is a 1D float vector
        u_des = np.asarray(u_des, dtype=float).reshape(-1)
        
        # Get dimension of control vector
        n = u_des.size  

        # Clean and normalize constraints
        A, b_ = [], []
        
        # For each a and b constraint term
        # output from the filter modules do
        for a, b in zip(a_list, b_list):
            a = np.asarray(a, dtype=float).reshape(n)
            b = float(b)
            
            # Remove invalid numerical constraints
            if not np.all(np.isfinite(a)) or not np.isfinite(b):
                continue

            # Skip zero or useless constraints
            norm = np.linalg.norm(a)
            if norm < drop_tol:
                continue

            # Normalize constraint row to improve numerical stability
            if normalize_rows:
                a = a / norm
                b = b / norm
            A.append(a)
            b_.append(b)

        m = len(A)

        # Problem: Minimize ||u - u_des||² 
        # Minimize (1/2)*uᵀPu + (-u_des)ᵀu + constant
        # With P = I
        P = sp.eye(n, format="csc")
        q = -u_des.copy()

        # Add CBF-provided quadratic penalties if present
        if P_extra is not None and q_extra is not None:
            P = P + (sp.csc_matrix(P_extra) if not isinstance(P_extra, sp.csc_matrix) else P_extra)
            q = q + q_extra

        # Build OSQP constraint matrices
        A_mat = np.stack(A, axis=0)
        b_arr = np.array(b_, dtype=float)

        # Set the constraint matrix
        Au = sp.csc_matrix(A_mat)    

        # Constraints formulation:
        # Lower bound: low <= Au
        low = b_arr

        # Upper bound : Au <= upp                    
        upp = np.full(m, np.inf)

        # Solve OSQP
        prob = osqp.OSQP()
        prob.setup(P=P, q=q, A=Au, l=low, u=upp, verbose=False, polish=False, eps_abs=1e-6, eps_rel=1e-6)
        res = prob.solve()

        # Check if OSQP successfully return a valid solution
        ok = (res.info.status_val in (1, 2)) and (res.x is not None)
        
        # If solver failed
        if not ok:
            # Falback to the desired and possibly unsafe input
            return u_des, np.zeros(m, dtype=float), False
        
        # Extract the optimal control input
        # Solver primal solution
        x = np.array(res.x, dtype=float)

        # Extract dual variables for each constraints
        # Solver Lagrangian multipliers
        # If solver did not return them, default to zero
        y = np.array(res.y, dtype=float) if res.y is not None else np.zeros(m, dtype=float)
        return x, y, True

    def apply(self, u_des, env):
        self.env = env

        # Copy nominal input 
        # First 7 components are the robot joint controls
        # Las component is the gripper open/close command
        u_act = np.array(u_des, dtype=float).copy()
        u_nom = u_act[:7]

        # Collect all constraints
        all_As, all_Bs = [], []
        row_start = 0
        groups = []

        # For each filter module
        for mod in self.modules:
            a_list, b_list = mod.constraints(self.env)
            
            # Get number of constraints by this mod
            k = len(a_list) 
            if k:
                # Append to the global constraints set
                all_As.extend(a_list)
                all_Bs.extend(b_list)

                # Record wher this module's block of constraints appears
                groups.append((mod.__class__.__name__, row_start, row_start + k))
                row_start += k

        # Collect optional quadratic objectives contribution
        P_extra_total = None
        q_extra_total = None

        # For each filter module
        for mod in self.modules:
            
            # Check if the module defines an objective term method
            obj = getattr(mod, "objective_terms", None)
            if obj is None:
                continue
            
            # Compute the additional objective term 
            out = mod.objective_terms(self.env)
            if out is None:
                continue
            
            # Add it as extra
            P_e, q_e = out
            P_extra_total = P_e if P_extra_total is None else P_extra_total + P_e
            q_extra_total = q_e if q_extra_total is None else q_extra_total + q_e

        # Solve full safety-filtered QP
        u_safe, _, ok = self._solve_qp(u_nom, all_As, all_Bs,
                                       P_extra=P_extra_total,
                                       q_extra=q_extra_total)

        # If solver succeeded, update the joint commands
        u_act[:7] = u_safe if ok else u_nom

        # Diagnostic information:
        # Contributions per CBF module
        efforts = {}
        if len(all_As) > 0:
            for name, _, _ in groups:
                efforts[name] = {"effort": 0.0}
        return u_act, efforts
