from argparse import ArgumentParser, RawTextHelpFormatter

import open3d as o3d
import torch
from torch.utils.data import DataLoader

from gradslam.datasets.realsense import Realsense
from gradslam.slam.icpslam import ICPSLAM
from gradslam import Pointclouds, RGBDImages

if __name__ == "__main__":
    # select device
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

    # load dataset
    loader = Realsense()

    pointclouds = Pointclouds(device=device)

    # SLAM
    slam = ICPSLAM(odom="gradicp", dsratio=4, device=device)
    initial_poses = torch.eye(4, device=device).view(1, 1, 4, 4)
    
    prev_frame = None
    s = 0
    while True:
        colors, depths, intrinsics = next(loader).to(device)
        if s == 0:
            live_frame = RGBDImages(colors, depths, intrinsics, initial_poses)
        pointclouds, live_frame.poses = slam.step(pointclouds, live_frame, prev_frame)
        prev_frame = live_frame
        s+=1
    pointclouds.plotly(0, max_num_points=15000).update_layout(autosize=False, width=600).show()