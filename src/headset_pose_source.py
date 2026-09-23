"""Single seam between the static-light (ceiling-lamp) exclusion feature's
geometry/map/runtime code and whatever currently supplies the headset's pose
in an absolute room frame. Today's only implementation wraps mocap
(src/mocap_data.py's world_pose) -- once SLAM/VIO is production-ready, a new
HeadsetPoseSource implementation can supply it instead, with zero changes
required in src/static_light_geometry.py, src/static_light_map.py, or
src/static_light_runtime.py.
"""
from typing import Optional

from src.mocap_data import DeviceMocap, world_pose
from src.transformations import Transform


class HeadsetPoseSource:
    def room_pose_at(self, frame_ts_ns: int) -> Optional[Transform]:
        """T_room_headsetImu at frame_ts_ns, or None if unavailable this frame
        (e.g. a mocap coverage gap). Callers must skip the frame -- never
        fabricate a pose."""
        raise NotImplementedError


class MocapHeadsetPoseSource(HeadsetPoseSource):
    """Wraps a headset DeviceMocap. The only file in this feature that imports
    DeviceMocap/world_pose directly -- everything downstream depends only on
    HeadsetPoseSource's interface."""

    def __init__(self, headset_mocap: DeviceMocap):
        self._headset_mocap = headset_mocap

    def room_pose_at(self, frame_ts_ns: int) -> Optional[Transform]:
        return world_pose(self._headset_mocap, frame_ts_ns)
