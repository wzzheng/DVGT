import argparse
from einops import rearrange
import numpy as np
import viser
import viser.transforms as viser_tf
from time import sleep
import torch
import numpy as np
import glob
from typing import Tuple, Union
from numbers import Number
import os
try:
    import onnxruntime
except ImportError:
    print("onnxruntime not found. Sky segmentation may not work.")
import cv2

from dvgt.models.dvgt import DVGT
from dvgt.utils.load_fn import load_and_preprocess_images
from iopath.common.file_io import g_pathmgr
from dvgt.utils.geometry import convert_point_in_ego_0_to_ray_depth_in_ego_n
from dvgt.utils.pose_enc import pose_encoding_to_ego_pose
from visual_util import segment_sky, download_file_from_url

def apply_sky_segmentation(conf: np.ndarray, images: np.ndarray) -> np.ndarray:
    """
    Apply sky segmentation to confidence scores.

    Args:
        conf (np.ndarray): Confidence scores with shape (T, V, H, W)
        images (np.ndarray): Image input model with shape (T, V, H, W, 3)

    Returns:
        np.ndarray: Updated confidence scores with sky regions masked out
    """
    T, V, H, W, _ = images.shape

    # Download skyseg.onnx if it doesn't exist
    if not os.path.exists("skyseg.onnx"):
        print("Downloading skyseg.onnx...")
        download_file_from_url("https://huggingface.co/JianyuanWang/skyseg/resolve/main/skyseg.onnx", "skyseg.onnx")

    skyseg_session = onnxruntime.InferenceSession("skyseg.onnx")
    sky_mask_list = []

    print("Generating sky masks...")
    for t in range(T):
        for v in range(V):
            sky_mask = segment_sky(images[t, v], skyseg_session)

            # Resize mask to match H×W if needed
            if sky_mask.shape[0] != H or sky_mask.shape[1] != W:
                sky_mask = cv2.resize(sky_mask, (W, H))

            sky_mask_list.append(sky_mask)

    # Convert list to numpy array with shape TVHW
    sky_mask_array = np.array(sky_mask_list).reshape(T, V, H, W)
    # Apply sky mask to confidence scores
    sky_mask_binary = (sky_mask_array > 0.1)
    # conf = conf * sky_mask_binary

    print("Sky segmentation applied successfully")
    return sky_mask_binary

def process_and_filter_points(points, colors, mask, max_depth, downsample_ratio):
    if max_depth > 0:
        print(f"Truncating points beyond max depth of {max_depth}m...")
        depth = np.linalg.norm(points, axis=-1)
        depth_mask = depth <= max_depth
        mask = mask & depth_mask

    if mask is not None:
        # Reshape mask from (T, V, H, W) to match points (N_points)
        mask_flat = mask.reshape(-1)
        # Reshape points/colors to (N_points, 3)
        points_flat = points.reshape(-1, 3)
        colors_flat = colors.reshape(-1, 3)
        
        points = points_flat[mask_flat]
        colors = colors_flat[mask_flat]
    else:
        # if mask is None，still needs flatten
        points = points.reshape(-1, 3)
        colors = colors.reshape(-1, 3)

    if 0 < downsample_ratio < 1.0:
        num_points = points.shape[0]
        target_num_points = int(num_points * downsample_ratio)
        if target_num_points < num_points and target_num_points > 0:
            print(f"Downsampling from {num_points} to {target_num_points} points...")
            indices = np.random.choice(num_points, target_num_points, replace=False)
            points = points[indices]
            colors = colors[indices]

    return points, colors

def center_data(points, poses):
    """
    Translate the point cloud and poses to a new coordinate system 
    where the point cloud's centroid is the origin.
    """
    if points.shape[0] == 0:
        return points, poses, np.zeros(3)
        
    center = np.mean(points, axis=0)
    points_centered = points - center
    
    poses_centered = poses.copy()
    # poses 是 (T, 3, 4)，中心点加在 t (最后一列)上
    poses_centered[..., -1] -= center
    
    print(f"Data centered. Original center was at {center}")
    return points_centered, poses_centered, center

def visualize_ego_poses(server, poses, frustum_scale=0.2, name_prefix="ego_pose"):
    print(f"Visualizing {len(poses)} ego poses...")
    for i, pose_3x4 in enumerate(poses):
        pose_4x4 = np.eye(4)
        pose_4x4[:3, :] = pose_3x4
        
        # T_world_camera = viser_tf.SE3.from_matrix(pose_3x4)
        T_world_camera = viser_tf.SE3.from_matrix(pose_4x4)
        
        server.scene.add_camera_frustum(
            f"/{name_prefix}/{i}",
            fov=np.pi / 4,  # set by default
            aspect=1.77,    # set by default
            scale=frustum_scale,
            wxyz=T_world_camera.rotation().wxyz,
            position=T_world_camera.translation(),
            color=(255, 100, 100)
        )

def angle_diff_vec3_numpy(v1: np.ndarray, v2: np.ndarray, eps: float = 1e-12):
    """
    Compute angle difference between 3D vectors using NumPy.
    """
    return np.arctan2(
        np.linalg.norm(np.cross(v1, v2, axis=-1), axis=-1) + eps, (v1 * v2).sum(axis=-1)
    )

def points_to_normals(
    point: np.ndarray, mask: np.ndarray = None, edge_threshold: float = None
) -> np.ndarray:
    """
    Calculate normal map from point map. Value range is [-1, 1].
    """
    height, width = point.shape[-3:-1]
    has_mask = mask is not None

    if mask is None:
        mask = np.ones_like(point[..., 0], dtype=bool)
    mask_pad = np.zeros((height + 2, width + 2), dtype=bool)
    mask_pad[1:-1, 1:-1] = mask
    mask = mask_pad

    pts = np.zeros((height + 2, width + 2, 3), dtype=point.dtype)
    pts[1:-1, 1:-1, :] = point
    up = pts[:-2, 1:-1, :] - pts[1:-1, 1:-1, :]
    left = pts[1:-1, :-2, :] - pts[1:-1, 1:-1, :]
    down = pts[2:, 1:-1, :] - pts[1:-1, 1:-1, :]
    right = pts[1:-1, 2:, :] - pts[1:-1, 1:-1, :]
    normal = np.stack(
        [
            np.cross(up, left, axis=-1),
            np.cross(left, down, axis=-1),
            np.cross(down, right, axis=-1),
            np.cross(right, up, axis=-1),
        ]
    )
    normal = normal / (np.linalg.norm(normal, axis=-1, keepdims=True) + 1e-12)

    valid = (
        np.stack(
            [
                mask[:-2, 1:-1] & mask[1:-1, :-2],
                mask[1:-1, :-2] & mask[2:, 1:-1],
                mask[2:, 1:-1] & mask[1:-1, 2:],
                mask[1:-1, 2:] & mask[:-2, 1:-1],
            ]
        )
        & mask[None, 1:-1, 1:-1]
    )
    if edge_threshold is not None:
        view_angle = angle_diff_vec3_numpy(pts[None, 1:-1, 1:-1, :], normal)
        view_angle = np.minimum(view_angle, np.pi - view_angle)
        valid = valid & (view_angle < np.deg2rad(edge_threshold))

    normal = (normal * valid[..., None]).sum(axis=0)
    normal = normal / (np.linalg.norm(normal, axis=-1, keepdims=True) + 1e-12)

    if has_mask:
        normal_mask = valid.any(axis=0)
        normal = np.where(normal_mask[..., None], normal, 0)
        return normal, normal_mask
    else:
        return normal

def sliding_window_1d(x: np.ndarray, window_size: int, stride: int, axis: int = -1):
    """
    Create a sliding window view of the input array along a specified axis.
    """
    assert x.shape[axis] >= window_size, (
        f"kernel_size ({window_size}) is larger than axis_size ({x.shape[axis]})"
    )
    axis = axis % x.ndim
    shape = (
        *x.shape[:axis],
        (x.shape[axis] - window_size + 1 + stride -1) // stride,
        *x.shape[axis + 1 :],
        window_size,
    )
    
    axis_size = x.shape[axis]
    n_windows = (axis_size - window_size) // stride + 1
    
    shape = (
        *x.shape[:axis],
        n_windows,
        *x.shape[axis + 1 :],
        window_size,
    )

    strides = (
        *x.strides[:axis],
        stride * x.strides[axis],
        *x.strides[axis + 1 :],
        x.strides[axis],
    )
    x_sliding = np.lib.stride_tricks.as_strided(x, shape=shape, strides=strides)
    return x_sliding

def sliding_window_nd(
    x: np.ndarray,
    window_size: Tuple[int, ...],
    stride: Tuple[int, ...],
    axis: Tuple[int, ...],
) -> np.ndarray:
    """
    Create sliding windows along multiple dimensions of the input array.
    """
    axis = [axis[i] % x.ndim for i in range(len(axis))]
    for i in range(len(axis)):
        x = sliding_window_1d(x, window_size[i], stride[i], axis[i])
    return x

def sliding_window_2d(
    x: np.ndarray,
    window_size: Union[int, Tuple[int, int]],
    stride: Union[int, Tuple[int, int]],
    axis: Tuple[int, int] = (-2, -1),
) -> np.ndarray:
    """
    Create 2D sliding windows over the input array.
    """
    if isinstance(window_size, int):
        window_size = (window_size, window_size)
    if isinstance(stride, int):
        stride = (stride, stride)
    return sliding_window_nd(x, window_size, stride, axis)

def max_pool_1d(
    x: np.ndarray, kernel_size: int, stride: int, padding: int = 0, axis: int = -1
):
    """
    Perform 1D max pooling on the input array.
    """
    axis = axis % x.ndim
    if padding > 0:
        fill_value = np.nan if x.dtype.kind == "f" else np.iinfo(x.dtype).min
        pad_shape = list(x.shape)
        pad_shape[axis] = padding
        padding_arr = np.full(
            tuple(pad_shape),
            fill_value=fill_value,
            dtype=x.dtype,
        )
        x = np.concatenate([padding_arr, x, padding_arr], axis=axis)
    
    a_sliding = sliding_window_1d(x, kernel_size, stride, axis)
    max_pool = np.nanmax(a_sliding, axis=-1)
    return max_pool

def max_pool_nd(
    x: np.ndarray,
    kernel_size: Tuple[int, ...],
    stride: Tuple[int, ...],
    padding: Tuple[int, ...],
    axis: Tuple[int, ...],
) -> np.ndarray:
    """
    Perform N-dimensional max pooling on the input array.
    """
    for i in range(len(axis)):
        x = max_pool_1d(x, kernel_size[i], stride[i], padding[i], axis[i])
    return x

def max_pool_2d(
    x: np.ndarray,
    kernel_size: Union[int, Tuple[int, int]],
    stride: Union[int, Tuple[int, int]],
    padding: Union[int, Tuple[int, int]],
    axis: Tuple[int, int] = (-2, -1),
):
    """
    Perform 2D max pooling on the input array.
    """
    if isinstance(kernel_size, Number):
        kernel_size = (kernel_size, kernel_size)
    if isinstance(stride, Number):
        stride = (stride, stride)
    if isinstance(padding, Number):
        padding = (padding, padding)
    axis = tuple(axis)
    return max_pool_nd(x, kernel_size, stride, padding, axis)

def depth_edge(
    depth: np.ndarray,
    atol: float = None,
    rtol: float = None,
    kernel_size: int = 3,
    mask: np.ndarray = None,
) -> np.ndarray:
    """
    Compute the edge mask from depth map.
    """
    if mask is None:
        diff = max_pool_2d(
            depth, kernel_size, stride=1, padding=kernel_size // 2
        ) + max_pool_2d(-depth, kernel_size, stride=1, padding=kernel_size // 2)
    else:
        diff = max_pool_2d(
            np.where(mask, depth, -np.inf),
            kernel_size,
            stride=1,
            padding=kernel_size // 2,
        ) + max_pool_2d(
            np.where(mask, -depth, -np.inf),
            kernel_size,
            stride=1,
            padding=kernel_size // 2,
        )

    edge = np.zeros_like(depth, dtype=bool)
    if atol is not None:
        edge |= diff > atol

    if rtol is not None:
        valid_depth = np.where(depth > 1e-6, depth, 1e-6)
        edge |= diff / valid_depth > rtol
    return edge

def normals_edge(
    normals: np.ndarray, tol: float, kernel_size: int = 3, mask: np.ndarray = None
) -> np.ndarray:
    """
    Compute the edge mask from normal map.
    """
    assert normals.ndim >= 3 and normals.shape[-1] == 3, (
        "normal should be of shape (..., height, width, 3)"
    )
    normals = normals / (np.linalg.norm(normals, axis=-1, keepdims=True) + 1e-12)

    padding = kernel_size // 2
    
    pad_width = [(0, 0)] * normals.ndim
    pad_width[normals.ndim - 3] = (padding, padding) # H
    pad_width[normals.ndim - 2] = (padding, padding) # W
    
    normals_padded = np.pad(
        normals,
        pad_width,
        mode="edge",
    )

    normals_window = sliding_window_2d(
        normals_padded,
        window_size=kernel_size,
        stride=1,
        axis=(-3, -2), # H, W axes
    )
    
    # normals shape: (H, W, 3) -> (H, W, 3, 1, 1)
    # normals_window shape: (H, W, 3, K, K)
    normals_expanded = normals[..., None, None]
    
    # (H, W, K, K)
    dot_prod = (normals_expanded * normals_window).sum(axis=-3)
    # clip to prevent arccos domain errors
    dot_prod_clipped = np.clip(dot_prod, -1.0, 1.0)
    angle_diff = np.arccos(dot_prod_clipped)

    if mask is not None:
        mask_pad_width = [(0, 0)] * mask.ndim
        mask_pad_width[mask.ndim - 2] = (padding, padding) # H
        mask_pad_width[mask.ndim - 1] = (padding, padding) # W
        
        mask_padded = np.pad(
            mask,
            mask_pad_width,
            mode="edge",
        )
        
        mask_window = sliding_window_2d(
            mask_padded,
            window_size=kernel_size,
            stride=1,
            axis=(-2, -1), # H, W axes
        )
        # angle_diff (H, W, K, K), mask_window (H, W, K, K)
        angle_diff = np.where(
            mask_window,
            angle_diff,
            0,
        )
        
    angle_diff = angle_diff.max(axis=(-2, -1))

    # The original implementation seems to have an extra max_pool, which might be for dilating the edges.
    # Replicating it here.
    angle_diff = max_pool_2d(
        angle_diff, kernel_size, stride=1, padding=kernel_size // 2
    )
    edge = angle_diff > np.deg2rad(tol)
    return edge

def visualize_pred(predictions, args):
    B, T, V, H, W, _ = predictions['world_points'].shape
    device = predictions['world_points'].device
    assert B == 1, "Only batch size = 1 is supported for the visualization."

    pred_ego_n_to_ego_0 = pose_encoding_to_ego_pose(predictions['ego_pose_enc'])

    pred_points = predictions['world_points']
    pred_ray_depth_in_ego_n = convert_point_in_ego_0_to_ray_depth_in_ego_n(pred_points, pred_ego_n_to_ego_0)

    # Squeeze the batch dimension for visualization.
    pred_points = pred_points[0].cpu().numpy()
    pred_points_conf = predictions['world_points_conf'][0].cpu().numpy()
    pred_depth = pred_ray_depth_in_ego_n[0].cpu().numpy()       # for depth edge mask
    pred_ego_poses = pred_ego_n_to_ego_0[0].cpu().numpy()
    images = rearrange(predictions['images'][0].cpu().numpy(), 't v c h w -> t v h w c') * 255   # T, V, H, W, 3
    images = images.astype(np.uint8)

    # construct the combined mask
    combined_mask = np.ones((T, V, H, W), dtype=bool)   

    if args.conf_threshold > 0:
        cutoff_value = np.percentile(pred_points_conf, args.conf_threshold)
        conf_mask = pred_points_conf >= cutoff_value
        combined_mask &= conf_mask

    if args.mask_sky:
        sky_mask = apply_sky_segmentation(pred_points_conf, images) 
        combined_mask &= sky_mask
    
    if args.use_edge_masks:
        # Applying edge masks
        edge_mask = np.ones_like(combined_mask)

        for t_idx in range(T):
            for v_idx in range(V):
                frame_pts = pred_points[t_idx, v_idx]  # (H, W, 3)
                frame_depth = pred_depth[t_idx, v_idx] # (H, W)
                frame_base_mask = combined_mask[t_idx, v_idx] # (H, W)

                # 1. Depth Edge Mask
                depth_edges = depth_edge(
                    frame_depth, 
                    rtol=args.edge_depth_rtol, 
                    mask=frame_base_mask
                )
                
                # 2. Normal Edge Mask
                normals, normals_mask = points_to_normals(
                    frame_pts, mask=frame_base_mask
                )
                normal_edges = normals_edge(
                    normals, tol=args.edge_normal_tol, mask=normals_mask
                )

                edge_mask[t_idx, v_idx] = ~(depth_edges | normal_edges)
            
            combined_mask &= edge_mask

    points_final, colors_final = process_and_filter_points(
        pred_points, images, combined_mask, args.max_depth, args.downsample_ratio
    )
    
    points_centered, poses_centered, center = center_data(points_final, pred_ego_poses)

    return [(points_centered, colors_final, "pred_point_cloud")], poses_centered

def main():
    parser = argparse.ArgumentParser(description="Autonomous Driving Scene Point Cloud Visualizer")
    parser.add_argument(
        "--image_folder", type=str, default="examples/openscene_log-0104-scene-0007/", help="Path to folder containing images"
    )
    parser.add_argument(
        "--checkpoint_path", type=str, default="ckpt/open_ckpt.pt", help="Path to folder containing images"
    )
    parser.add_argument("--start_frame", type=int, default=16, help="The start frame in the example autonomous video.")
    parser.add_argument("--end_frame", type=int, default=20, help="The end frame in the example autonomous video.")

    parser.add_argument('--downsample_ratio', type=float, default=-1, help="Random downsample ratio (0.0 to 1.0). Default: -1 (no downsampling).")
    parser.add_argument('--max_depth', type=float, default=-1, help="Maximum depth of points to visualize in meters. Default: -1 (no truncation).")

    parser.add_argument('--no_ego', action='store_true', help="Disable ego pose visualization.")
    parser.add_argument('--use_edge_masks', action='store_true', help="Enable depth and normal edge masks.")
    parser.add_argument("--mask_sky", action="store_true", help="Apply sky segmentation to filter out sky points")

    parser.add_argument('--edge_depth_rtol', type=float, default=0.1, help="Relative tolerance (rtol) for depth edge detection.")
    parser.add_argument('--edge_normal_tol', type=float, default=50, help="Angle tolerance (degrees) for normal edge detection.")
    parser.add_argument("--conf_threshold", type=float, default=25.0, help="Initial percentage of low-confidence points to filter out")

    args = parser.parse_args()

    # load model
    device = "cuda" if torch.cuda.is_available() else "cpu"
    # bfloat16 is supported on Ampere GPUs (Compute Capability 8.0+) 
    dtype = torch.bfloat16 if torch.cuda.get_device_capability()[0] >= 8 else torch.float16

    # Initialize the model and load the pretrained weights.
    model = DVGT()
    with g_pathmgr.open(args.checkpoint_path, "rb") as f:
        checkpoint = torch.load(f, map_location="cpu")
    model.load_state_dict(checkpoint)
    model = model.to(device).eval()

    # Load and preprocess example images (replace with your own image paths)
    images = load_and_preprocess_images(args.image_folder, start_frame=args.start_frame, end_frame=args.end_frame).to(device)

    with torch.no_grad():
        with torch.amp.autocast(device, dtype=dtype):
            # Predict attributes including cameras, depth maps, and point maps.
            predictions = model(images)
    
    point_clouds_to_show, poses = visualize_pred(predictions, args)

    server = viser.ViserServer()
    print("\n--- Starting Viser Server ---")

    @server.on_client_connect
    def _(client: viser.ClientHandle):
        print(f"Client {client.client_id} connected.")
        
        with client.gui.add_folder("Camera Info"):
            gui_cam_wxyz = client.gui.add_text("Cam WXYZ (quat)", initial_value="...")
            gui_cam_pos = client.gui.add_text("Cam Position (xyz)", initial_value="...")

        @client.camera.on_update
        def _(camera: viser.CameraHandle):
            """Callback for camera updates."""
            gui_cam_wxyz.value = str(np.round(camera.wxyz, 3))
            gui_cam_pos.value = str(np.round(camera.position, 3))
        
        _(client.camera)

    with server.gui.add_folder("Controls"):
        point_size_slider = server.gui.add_slider(
            "Point Size",
            min=0.001,
            max=0.1,
            step=0.01,
            initial_value=0.01,
        )

    pc_handles = []
    for points, colors, name in point_clouds_to_show:
        if points.shape[0] == 0: continue
        print(f"point shape: {points.shape}")
        print(f"point mean: {np.mean(points, axis=0)}")
        handle = server.scene.add_point_cloud(
            name=f"/{name}",
            points=points,
            colors=colors,
            point_size=point_size_slider.value,
        )
        pc_handles.append(handle)

    @point_size_slider.on_update
    def _(_) -> None:
        """Update point size for all point clouds."""
        for handle in pc_handles:
            handle.point_size = point_size_slider.value

    if not args.no_ego:
        visualize_ego_poses(server, poses)

    print("\nViser server running. Open the link in your browser.")
    while True:
        sleep(1)

if __name__ == "__main__":
    main()