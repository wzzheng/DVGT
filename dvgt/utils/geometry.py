# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import os
import torch
import numpy as np
from typing import Dict, Union

from dvgt.dependency.distortion import apply_distortion, iterative_undistortion, single_undistortion


def unproject_depth_map_to_point_map(
    depth_map: np.ndarray, extrinsics_cam: np.ndarray, intrinsics_cam: np.ndarray
) -> np.ndarray:
    """
    Unproject a batch of depth maps to 3D world coordinates.

    Args:
        depth_map (np.ndarray): Batch of depth maps of shape (S, H, W, 1) or (S, H, W)
        extrinsics_cam (np.ndarray): Batch of camera extrinsic matrices of shape (S, 3, 4)
        intrinsics_cam (np.ndarray): Batch of camera intrinsic matrices of shape (S, 3, 3)

    Returns:
        np.ndarray: Batch of 3D world coordinates of shape (S, H, W, 3)
    """
    if isinstance(depth_map, torch.Tensor):
        depth_map = depth_map.cpu().numpy()
    if isinstance(extrinsics_cam, torch.Tensor):
        extrinsics_cam = extrinsics_cam.cpu().numpy()
    if isinstance(intrinsics_cam, torch.Tensor):
        intrinsics_cam = intrinsics_cam.cpu().numpy()

    world_points_list = []
    for frame_idx in range(depth_map.shape[0]):
        cur_world_points, _, _ = depth_to_world_coords_points(
            depth_map[frame_idx].squeeze(-1), extrinsics_cam[frame_idx], intrinsics_cam[frame_idx]
        )
        world_points_list.append(cur_world_points)
    world_points_array = np.stack(world_points_list, axis=0)

    return world_points_array


def depth_to_world_coords_points(
    depth_map: np.ndarray,
    extrinsic: np.ndarray,
    intrinsic: np.ndarray,
    eps=1e-8,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Convert a depth map to world coordinates.

    Args:
        depth_map (np.ndarray): Depth map of shape (H, W).
        intrinsic (np.ndarray): Camera intrinsic matrix of shape (3, 3).
        extrinsic (np.ndarray): Camera extrinsic matrix of shape (3, 4). OpenCV camera coordinate convention, cam from world.

    Returns:
        tuple[np.ndarray, np.ndarray]: 
            World coordinates (H, W, 3)
            cam coordinates (H, W, 3)
            valid depth mask (H, W)
    """
    if depth_map is None:
        return None, None, None

    # Valid depth mask
    point_mask = depth_map > eps

    # Convert depth map to camera coordinates
    cam_coords_points = depth_to_cam_coords_points(depth_map, intrinsic)

    # Multiply with the inverse of extrinsic matrix to transform to world coordinates
    # extrinsic_inv is 4x4 (note closed_form_inverse_OpenCV is batched, the output is (N, 4, 4))
    cam_to_world_extrinsic = closed_form_inverse_se3(extrinsic)

    R_cam_to_world = cam_to_world_extrinsic[:3, :3]
    t_cam_to_world = cam_to_world_extrinsic[:3, 3]

    # Apply the rotation and translation to the camera coordinates
    world_coords_points = np.dot(cam_coords_points, R_cam_to_world.T) + t_cam_to_world  # HxWx3, 3x3 -> HxWx3

    return world_coords_points, cam_coords_points, point_mask


def depth_to_cam_coords_points(depth_map: np.ndarray, intrinsic: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """
    Convert a depth map to camera coordinates.

    Args:
        depth_map (np.ndarray): Depth map of shape (H, W).
        intrinsic (np.ndarray): Camera intrinsic matrix of shape (3, 3).

    Returns:
        tuple[np.ndarray, np.ndarray]: Camera coordinates (H, W, 3)
    """
    H, W = depth_map.shape
    assert intrinsic.shape == (3, 3), "Intrinsic matrix must be 3x3"
    assert intrinsic[0, 1] == 0 and intrinsic[1, 0] == 0, "Intrinsic matrix must have zero skew"

    # Intrinsic parameters
    fu, fv = intrinsic[0, 0], intrinsic[1, 1]
    cu, cv = intrinsic[0, 2], intrinsic[1, 2]

    # Generate grid of pixel coordinates
    u, v = np.meshgrid(np.arange(W), np.arange(H))

    # Unproject to camera coordinates
    x_cam = (u - cu) * depth_map / fu
    y_cam = (v - cv) * depth_map / fv
    z_cam = depth_map

    # Stack to form camera coordinates
    cam_coords = np.stack((x_cam, y_cam, z_cam), axis=-1).astype(np.float32)

    return cam_coords


def closed_form_inverse_se3(se3: torch.Tensor or np.ndarray) -> torch.Tensor or np.ndarray:
    """
    以闭式解计算一批SE(3)矩阵的逆。
    这个函数支持任意批次维度 (...) 以及单个矩阵。

    Args:
        se3: SE(3) 变换矩阵，形状为 (..., 4, 4) 或 (..., 3, 4)。

    Returns:
        逆变换矩阵，与输入 se3 的类型、设备和批次维度相同，形状为 (..., 4, 4)。
        
    公式:
        如果 T_mat = [[R, t], [0, 1]]
        则 T_mat_inv = [[R.T, -R.T @ t], [0, 1]]
    """
    # 检查输入是 NumPy 数组还是 PyTorch 张量
    is_numpy = isinstance(se3, np.ndarray)
    
    # 为单个矩阵 (ndim=2) 临时增加一个批次维度，以便统一处理
    was_single = se3.ndim == 2
    if was_single:
        se3 = se3[None, ...]  # works for both numpy and torch

    # 验证最后两个维度
    if se3.shape[-2:] not in [(4, 4), (3, 4)]:
        raise ValueError(f"输入矩阵的最后两个维度必须是 (4, 4) 或 (3, 4)，但得到的是 {se3.shape}")

    R = se3[..., :3, :3]
    t = se3[..., :3, 3:4]

    if is_numpy:
        R_transposed = np.swapaxes(R, -2, -1)
    else:
        R_transposed = R.transpose(-2, -1)
    
    t_inverted = -R_transposed @ t

    batch_shape = se3.shape[:-2]
    
    if is_numpy:
        inverted_matrix = np.zeros((*batch_shape, 4, 4), dtype=se3.dtype)
    else:
        inverted_matrix = torch.zeros((*batch_shape, 4, 4), device=se3.device, dtype=se3.dtype)
    
    inverted_matrix[..., :3, :3] = R_transposed
    inverted_matrix[..., :3, 3:4] = t_inverted
    inverted_matrix[..., 3, 3] = 1.0

    # 5. 如果输入是单个矩阵，移除之前添加的批次维度
    if was_single:
        inverted_matrix = inverted_matrix[0]

    return inverted_matrix

def to_homogeneous(matrix: Union[torch.Tensor, np.ndarray]) -> Union[torch.Tensor, np.ndarray]:
    """
    将一个 (..., 3, 4) 的变换矩阵批量转换为 (..., 4, 4) 的齐次矩阵。
    """
    is_torch = isinstance(matrix, torch.Tensor)

    batch_dims = matrix.shape[:-2]
    
    if is_torch:
        last_row_expanded = torch.zeros((*batch_dims, 1, 4), dtype=matrix.dtype, device=matrix.device)
        last_row_expanded[..., 3] = 1.0
        return torch.cat([matrix, last_row_expanded], dim=-2)
    else:
        last_row_expanded = np.zeros((*batch_dims, 1, 4), dtype=matrix.dtype)
        last_row_expanded[..., 3] = 1.0
        return np.concatenate([matrix, last_row_expanded], axis=-2)

def convert_point_in_ego_0_to_ray_depth_in_ego_n(
    point_in_ego_0: torch.Tensor,   # [B, T, V, H, W, 3]
    ego_n_to_ego_0: torch.Tensor,   # [B, T, 3, 4]
) -> torch.Tensor:
    point_in_ego_0 = point_in_ego_0.to(torch.float64)
    ego_n_to_ego_0 = ego_n_to_ego_0.to(torch.float64)

    ego_0_to_ego_n = closed_form_inverse_se3(ego_n_to_ego_0)

    point_in_ego_n = point_in_ego_0 @ ego_0_to_ego_n[:, :, None, None, :3, :3].transpose(-1, -2) \
        + ego_0_to_ego_n[:, :, None, None, None, :3, 3]
    
    ray_depth = point_in_ego_n.norm(dim=-1, p=2)
    return ray_depth.to(torch.float32)

def project_world_points_to_camera_points_batch(world_points, cam_extrinsics):
    """
    Transforms 3D points to 2D using extrinsic and intrinsic parameters.
    Args:
        world_points (torch.Tensor): 3D points of shape BxSxHxWx3.
        cam_extrinsics (torch.Tensor): Extrinsic parameters of shape BxSx3x4.
    Returns:
    """
    # TODO: merge this into project_world_points_to_cam
    
    # device = world_points.device
    # with torch.autocast(device_type=device.type, enabled=False):
    ones = torch.ones_like(world_points[..., :1])  # shape: (B, S, H, W, 1)
    world_points_h = torch.cat([world_points, ones], dim=-1)  # shape: (B, S, H, W, 4)

    # extrinsics: (B, S, 3, 4) -> (B, S, 1, 1, 3, 4)
    extrinsics_exp = cam_extrinsics.unsqueeze(2).unsqueeze(3)

    # world_points_h: (B, S, H, W, 4) -> (B, S, H, W, 4, 1)
    world_points_h_exp = world_points_h.unsqueeze(-1)

    # Now perform the matrix multiplication
    # (B, S, 1, 1, 3, 4) @ (B, S, H, W, 4, 1) broadcasts to (B, S, H, W, 3, 1)
    camera_points = torch.matmul(extrinsics_exp, world_points_h_exp).squeeze(-1)

    return camera_points



def project_world_points_to_cam(
    world_points,
    cam_extrinsics,
    cam_intrinsics=None,
    distortion_params=None,
    default=0,
    only_points_cam=False,
):
    """
    Transforms 3D points to 2D using extrinsic and intrinsic parameters.
    Args:
        world_points (torch.Tensor): 3D points of shape Px3.
        cam_extrinsics (torch.Tensor): Extrinsic parameters of shape Bx3x4.
        cam_intrinsics (torch.Tensor): Intrinsic parameters of shape Bx3x3.
        distortion_params (torch.Tensor): Extra parameters of shape BxN, which is used for radial distortion.
    Returns:
        torch.Tensor: Transformed 2D points of shape BxNx2.
    """
    device = world_points.device
    # with torch.autocast(device_type=device.type, dtype=torch.double):
    with torch.autocast(device_type=device.type, enabled=False):
        N = world_points.shape[0]  # Number of points
        B = cam_extrinsics.shape[0]  # Batch size, i.e., number of cameras
        world_points_homogeneous = torch.cat(
            [world_points, torch.ones_like(world_points[..., 0:1])], dim=1
        )  # Nx4
        # Reshape for batch processing
        world_points_homogeneous = world_points_homogeneous.unsqueeze(0).expand(
            B, -1, -1
        )  # BxNx4

        # Step 1: Apply extrinsic parameters
        # Transform 3D points to camera coordinate system for all cameras
        cam_points = torch.bmm(
            cam_extrinsics, world_points_homogeneous.transpose(-1, -2)
        )

        if only_points_cam:
            return None, cam_points

        # Step 2: Apply intrinsic parameters and (optional) distortion
        image_points = img_from_cam(cam_intrinsics, cam_points, distortion_params, default=default)

        return image_points, cam_points



def img_from_cam(cam_intrinsics, cam_points, distortion_params=None, default=0.0):
    """
    Applies intrinsic parameters and optional distortion to the given 3D points.

    Args:
        cam_intrinsics (torch.Tensor): Intrinsic camera parameters of shape Bx3x3.
        cam_points (torch.Tensor): 3D points in camera coordinates of shape Bx3xN.
        distortion_params (torch.Tensor, optional): Distortion parameters of shape BxN, where N can be 1, 2, or 4.
        default (float, optional): Default value to replace NaNs in the output.

    Returns:
        pixel_coords (torch.Tensor): 2D points in pixel coordinates of shape BxNx2.
    """

    # Normalized device coordinates (NDC)
    cam_points = cam_points / cam_points[:, 2:3, :]
    ndc_xy = cam_points[:, :2, :]

    # Apply distortion if distortion_params are provided
    if distortion_params is not None:
        x_distorted, y_distorted = apply_distortion(distortion_params, ndc_xy[:, 0], ndc_xy[:, 1])
        distorted_xy = torch.stack([x_distorted, y_distorted], dim=1)
    else:
        distorted_xy = ndc_xy

    # Prepare cam_points for batch matrix multiplication
    cam_coords_homo = torch.cat(
        (distorted_xy, torch.ones_like(distorted_xy[:, :1, :])), dim=1
    )  # Bx3xN
    # Apply intrinsic parameters using batch matrix multiplication
    pixel_coords = torch.bmm(cam_intrinsics, cam_coords_homo)  # Bx3xN

    # Extract x and y coordinates
    pixel_coords = pixel_coords[:, :2, :]  # Bx2xN

    # Replace NaNs with default value
    pixel_coords = torch.nan_to_num(pixel_coords, nan=default)

    return pixel_coords.transpose(1, 2)  # BxNx2




def cam_from_img(pred_tracks, intrinsics, extra_params=None):
    """
    Normalize predicted tracks based on camera intrinsics.
    Args:
    intrinsics (torch.Tensor): The camera intrinsics tensor of shape [batch_size, 3, 3].
    pred_tracks (torch.Tensor): The predicted tracks tensor of shape [batch_size, num_tracks, 2].
    extra_params (torch.Tensor, optional): Distortion parameters of shape BxN, where N can be 1, 2, or 4.
    Returns:
    torch.Tensor: Normalized tracks tensor.
    """

    # We don't want to do intrinsics_inv = torch.inverse(intrinsics) here
    # otherwise we can use something like
    #     tracks_normalized_homo = torch.bmm(pred_tracks_homo, intrinsics_inv.transpose(1, 2))

    principal_point = intrinsics[:, [0, 1], [2, 2]].unsqueeze(-2)
    focal_length = intrinsics[:, [0, 1], [0, 1]].unsqueeze(-2)
    tracks_normalized = (pred_tracks - principal_point) / focal_length

    if extra_params is not None:
        # Apply iterative undistortion
        try:
            tracks_normalized = iterative_undistortion(
                extra_params, tracks_normalized
            )
        except:
            tracks_normalized = single_undistortion(
                extra_params, tracks_normalized
            )

    return tracks_normalized

def project_points_in_ego_first_to_depth(
    points_in_ego_first: torch.Tensor,
    ego_first_to_world: torch.Tensor,
    extrinsics: torch.Tensor
) -> torch.Tensor:
    """
    将“第一帧自车坐标系”下的3D点投影回相机深度图。用于可视化

    这个函数执行以下两步转换：
    1. points_in_ego_first -> points_in_world: 使用 ego_first_to_world 变换。
    2. points_in_world -> points_in_camera: 使用相机外参 extrinsics 变换。
    最后，提取相机坐标系下的 Z 轴坐标作为深度。

    Args:
        points_in_ego_first (torch.Tensor): (V, H, W, 3) "第一帧自车坐标系"下的3D点。
        ego_first_to_world (torch.Tensor): (3, 4) or (4, 4) 从第一帧自车到世界的变换矩阵。
        extrinsics (torch.Tensor): (V, 3, 4) or (V, 4, 4) V个相机的外参（world to cam）。

    Returns:
        torch.Tensor: (V, H, W) 生成的深度图。
    """
    points_in_ego_first = points_in_ego_first.to(torch.float64)
    ego_first_to_world = ego_first_to_world.to(torch.float64)
    extrinsics = extrinsics.to(torch.float64)
    
    V, H, W, _ = points_in_ego_first.shape
    device = points_in_ego_first.device

    # --- 步骤 1: 从 "第一帧自车" 坐标系转换到世界坐标系 ---
    # 分离 ego_first_to_world 的旋转 R 和平移 T
    R_e1_w = ego_first_to_world[:3, :3]  # (3, 3)
    t_e1_w = ego_first_to_world[:3, 3]   # (3,)

    # 应用变换: P_world = P_ego_first * R^T + T
    # (V, H, W, 3) @ (3, 3) -> (V, H, W, 3)
    # (3,) 会被广播到 (V, H, W, 3)
    points_world = points_in_ego_first @ R_e1_w.T + t_e1_w

    # --- 步骤 2: 从世界坐标系转换到每个相机的坐标系 ---
    # 分离相机外参的旋转 R 和平移 T
    R_w_c = extrinsics[:, :3, :3]  # (V, 3, 3)
    t_w_c = extrinsics[:, :3, 3]   # (V, 3)

    points_cam = []
    for point, R, T in zip(points_world, R_w_c, t_w_c):
        points_cam.append(
            point @ R.T + T
        )
    points_cam = torch.stack(points_cam)
    depth = points_cam[..., 2]    

    depth[depth < 0] = 0

    return depth.float()