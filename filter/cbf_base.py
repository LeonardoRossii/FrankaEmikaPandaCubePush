from typing import Optional, Tuple, Sequence, Any

class CBFModule:
    """
    Base class for Control Barrier Function (CBF) modules.
    Implement:
      - constraints(env) -> (list[np.ndarray], list[float])
      - objective_terms(env) -> Optional[Tuple[np.ndarray, np.ndarray]]
    """
    def constraints(self, env: Any) -> Tuple[Sequence[Any], Sequence[float]]:
        raise NotImplementedError

    def objective_terms(self, env: Any) -> Optional[Tuple[Any, Any]]:
        return None
