
from typing import Any

import rerun as rr
import rerun.blueprint as rrb
import torch

from lerobot.common.robot_devices.robots.manipulator import ManipulatorRobot

def target_state_action_blueprint(robot: ManipulatorRobot, root="") -> rrb.Blueprint:
    """
    Returns blueprint that will show a TimeSeriesView for each state/action pair.
    Looks good if the action is the target state and both are list of floats.
    """
    state_len = sum(len(arm.motors) for arm in robot.follower_arms.values())
    return rrb.Blueprint(
        rrb.Horizontal(
            rrb.Grid(
                *(
                    rrb.Spatial2DView(contents=[f"/observation/images/{camera_name}"])
                    for camera_name in robot.cameras
                )
            ),
            rrb.Vertical(
                *(
                    rrb.TimeSeriesView(contents=[f"/action/{i}", f"observation/state/{i}"])
                    for i in range(state_len)
                ),
                rrb.TimeSeriesView(contents=['/control_info/**'])
            ),
            column_shares=[3, 1]
        ),
        auto_layout=False,
        auto_space_views=False,
        collapse_panels=True,
    )

def _log_value(value, rr_path="") -> None:
    if "images" in rr_path:
        rr.log(rr_path, rr.Image(value))
    elif type(value) is dict:
        log_observation(value, root=rr_path)
    elif (type(value) is torch.Tensor and value.dim() == 0) or type(value) is float or type(value) is int:
        rr.log(rr_path, rr.Scalar(value))
    elif (type(value) is torch.Tensor and value.dim() == 1) or type(value) is list:
        for i, val in enumerate(value):
            rr.log(f"{rr_path}/{i}", rr.Scalar(val))

def log_observation(observation: dict[str, Any], root="") -> None:
    for key, value in observation.items():
        rr_path = f'{root}/{key.replace('.', '/')}'
        _log_value(value, rr_path)
