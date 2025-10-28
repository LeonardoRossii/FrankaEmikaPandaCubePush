from dataclasses import dataclass, field
from pathlib import Path
import json
import numpy as np
from typing import List, Dict, Any

@dataclass
class RolloutMetrics:
    log_every: int = 10
    cube_to_goal_dist: List[float] = field(default_factory=list)
    cube_to_bound_dist: List[float] = field(default_factory=list)
    eef_to_cube_dist: List[float] = field(default_factory=list)
    push_alignment_cos: List[float] = field(default_factory=list)
    drops: List[int] = field(default_factory=list)

    def log_step(self, t: int, obs: Dict[str, Any]) -> None:

        if t % self.log_every != 0:
            return
    
        cube_to_goal = obs.get("cube_to_goal", None)
        if cube_to_goal is not None:
            cube_to_goal = np.asarray(cube_to_goal)
            self.cube_to_goal_dist.append(float(np.linalg.norm(cube_to_goal)))

        cube_to_bound_dist = obs.get("cube_to_bound_dist", None)
        if cube_to_bound_dist is not None:
            self.cube_to_bound_dist.append(float(cube_to_bound_dist))

        eef_to_cube = obs.get("eef_to_cube", None)
        if eef_to_cube is not None:
            eef_to_cube = np.asarray(eef_to_cube)
            self.eef_to_cube_dist.append(float(np.linalg.norm(eef_to_cube)))

        # Push alignment: how well the EEF is positioned behind the cube relative to the goal direction (XY plane)
        # 1.0 means perfectly behind (ideal for pushing), 0.0 perpendicular, negative means in front of cube (blocking).
        if (eef_to_cube is not None) and (cube_to_goal is not None):
            u = np.asarray(eef_to_cube[:2])
            v = np.asarray(cube_to_goal[:2])
            nu = np.linalg.norm(u)
            nv = np.linalg.norm(v)
            if nu > 1e-8 and nv > 1e-8:
                cos = float(np.clip(np.dot(u, -v) / (nu * nv), -1.0, 1.0))
            else:
                cos = 0.0
            self.push_alignment_cos.append(cos)

        drop = obs.get("cube_drop", None)
        if drop is not None:
            self.drops.append(int(bool(drop)))

    def to_dict(self, accomplished: bool) -> Dict[str, Any]:
        return {
            "episode_cube_to_goal": self.cube_to_goal_dist,
            "episode_eef_to_cube": self.eef_to_cube_dist,
            "episode_cube_to_bound": self.cube_to_bound_dist,
            "episode_push_alignment_cos": self.push_alignment_cos,
            "episode_drops": self.drops,
            "task_accomplished": bool(accomplished),
        }

    def save(self, path: str, accomplished: bool) -> None:
        p = Path(path)
        p.parent.mkdir(parents=True, exist_ok=True)
        with p.open("w") as f:
            json.dump(self.to_dict(accomplished), f, indent=2)