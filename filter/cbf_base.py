from typing import Optional, Tuple, Sequence, Any

class CBFModule:
    """
    Base class for Control Barrier Function (CBF) modules.
    A CBFModule defines safety constraints and optionally cost/objective terms
    that can be incorporated into a controller, such as a Quadratic Program (QP)
    used for safe control.
    """
    
    def constraints(self, env: Any) -> Tuple[Sequence[Any], Sequence[float]]:
        raise NotImplementedError

    def objective_terms(self, env: Any) -> Optional[Tuple[Any, Any]]:
        return None
