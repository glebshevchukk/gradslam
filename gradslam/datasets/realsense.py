import pyrealsense2 as rs
import numpy as np

class Realsense():
    def __init__(self):
        self.pipeline = rs.pipeline()
        config = rs.config()
        config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)
        config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)

        # Start streaming
        profile = self.pipeline.start(config)
        depth_sensor = profile.get_device().first_depth_sensor()
        align_to = rs.stream.color
        self.align = rs.align(align_to)
        self.intrinsics = profile.as_video_stream_profile().get_intrinsics()
        print("Intrinsics")
        print(self.intrinsics)

    def __next__(self):
        frames = self.pipeline.wait_for_frames()
        # Align the depth frame to color frame
        aligned_frames = self.align.process(frames)

        # Get aligned frames
        aligned_depth_frame = aligned_frames.get_depth_frame() # aligned_depth_frame is a 640x480 depth image
        color_frame = aligned_frames.get_color_frame()

        # Validate that both frames are valid
        if not aligned_depth_frame or not color_frame:
            print("Either color or depth frame is invalid, skipping")
            return

        depth_image = np.asanyarray(aligned_depth_frame.get_data())
        color_image = np.asanyarray(color_frame.get_data())


        return color_image, depth_image, self.intrinsics


