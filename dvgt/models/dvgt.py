import torch
import torch.nn as nn
from huggingface_hub import PyTorchModelHubMixin  # used for model hub
from typing import List

from .aggregator import Aggregator
from dvgt.heads.ego_pose_head import EgoPoseHead
from dvgt.heads.dpt_head import DPTHead

class DVGT(nn.Module, PyTorchModelHubMixin):
    def __init__(
        self, 
        patch_size=16, 
        embed_dim=1024, 
        enable_ego_pose=True, 
        enable_point=True, 
        frames_chunk_size=None, # DPT head: Number of frames to process in each chunk.
    ):
        super().__init__()

        self.frames_chunk_size = frames_chunk_size

        dim_in = 3 * embed_dim

        self.aggregator = Aggregator(patch_size=patch_size, embed_dim=embed_dim)

        self.ego_pose_head = EgoPoseHead(dim_in=dim_in) if enable_ego_pose else None
        self.point_head = DPTHead(dim_in=dim_in, patch_size=patch_size) if enable_point else None

    def forward(self, images: torch.Tensor):
        """
        Forward pass of the DVGT model.

        Args:
            images (torch.Tensor): Input images with shape [T, V, 3, H, W] or [B, T, V, 3, H, W], in range [0, 1].
                B: batch size, T: num_frames, V: views_per_frame, 3: RGB channels, H: height, W: width

        Returns:
            dict: A dictionary containing the following predictions:
                - ego_pose_enc (torch.Tensor): Ego pose encoding with shape [B, T, V, 7] (from the last iteration)
                - world_points (torch.Tensor): world_points (torch.Tensor): 3D coordinates for each pixel relative to the ego-frame of the first timestep, of shape [B, T, V, H, W, 3].
                - world_points_conf (torch.Tensor): Confidence scores for world points with shape [B, T, V, H, W]
                - images (torch.Tensor): Original input images, preserved for visualization
        """        
        # If without batch dimension, add it
        if len(images.shape) == 5:
            images = images.unsqueeze(0)
            
        aggregated_tokens_list, patch_start_idx = self.aggregator(images)

        predictions = {}

        with torch.amp.autocast(device_type=images.device.type, enabled=False):
            if self.ego_pose_head is not None:
                ego_pose_enc_list = self.ego_pose_head(aggregated_tokens_list)
                predictions["ego_pose_enc"] = ego_pose_enc_list[-1]  # pose encoding of the last iteration
                predictions["ego_pose_enc_list"] = ego_pose_enc_list

            if self.point_head is not None:
                pts3d, pts3d_conf = self.point_head(
                    aggregated_tokens_list, images=images, patch_start_idx=patch_start_idx, frames_chunk_size=self.frames_chunk_size
                )
                predictions["world_points"] = pts3d
                predictions["world_points_conf"] = pts3d_conf

        if not self.training:
            predictions["images"] = images  # store the images for visualization during inference

        return predictions

