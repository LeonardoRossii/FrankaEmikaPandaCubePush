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
    eef_table_clearance: List[float] = field(default_factory=list)
    drops: List[int] = field(default_factory=list)

    def log_step(self, t: int, obs: Dict[str, Any]) -> None:

        if t % self.log_every != 0:
            return
    
        cube_to_goal = obs.get("cube_to_goal", None)
        if cube_to_goal is not None:
            self.cube_to_goal_dist.append(float(np.linalg.norm(cube_to_goal)))

        cube_to_bound_dist = obs.get("cube_to_bound_dist", None)
        if cube_to_bound_dist is not None:
            self.cube_to_bound_dist.append(float(cube_to_bound_dist))

        eef_to_cube = obs.get("eef_to_cube", None)
        if eef_to_cube is not None:
            self.eef_to_cube_dist.append(float(np.linalg.norm(eef_to_cube)))

        # End-effector clearance from table (approx.): z component of eef_to_goal.
        # goal_z is set ~ table_z + 0.001 in the observable, so positive values mean above table.
        eef_to_goal = obs.get("eef_to_goal", None)
        if eef_to_goal is not None:
            self.eef_table_clearance.append(float(eef_to_goal[2]))

        drop = obs.get("cube_drop", None)
        if drop is not None:
            self.drops.append(int(bool(drop)))

    def to_dict(self, accomplished: bool) -> Dict[str, Any]:
        return {
            "episode_cube_to_goal": self.cube_to_goal_dist,
            "episode_eef_to_cube": self.eef_to_cube_dist,
            "episode_cube_to_bound": self.cube_to_bound_dist,
            "episode_eef_table_clearance": self.eef_table_clearance,
            "episode_drops": self.drops,
            "task_accomplished": bool(accomplished),
        }

    def save(self, path: str, accomplished: bool) -> None:
        p = Path(path)
        p.parent.mkdir(parents=True, exist_ok=True)
        with p.open("w") as f:
            json.dump(self.to_dict(accomplished), f, indent=2)