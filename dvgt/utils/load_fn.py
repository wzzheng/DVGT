# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import torch
from PIL import Image
from torchvision import transforms as TF
import numpy as np
import os

def load_and_preprocess_images_square(image_path_list, target_size=1024):
    """
    Load and preprocess images by center padding to square and resizing to target size.
    Also returns the position information of original pixels after transformation.

    Args:
        image_path_list (list): List of paths to image files
        target_size (int, optional): Target size for both width and height. Defaults to 518.

    Returns:
        tuple: (
            torch.Tensor: Batched tensor of preprocessed images with shape (N, 3, target_size, target_size),
            torch.Tensor: Array of shape (N, 5) containing [x1, y1, x2, y2, width, height] for each image
        )

    Raises:
        ValueError: If the input list is empty
    """
    # Check for empty list
    if len(image_path_list) == 0:
        raise ValueError("At least 1 image is required")

    images = []
    original_coords = []  # Renamed from position_info to be more descriptive
    to_tensor = TF.ToTensor()

    for image_path in image_path_list:
        # Open image
        img = Image.open(image_path)

        # If there's an alpha channel, blend onto white background
        if img.mode == "RGBA":
            background = Image.new("RGBA", img.size, (255, 255, 255, 255))
            img = Image.alpha_composite(background, img)

        # Convert to RGB
        img = img.convert("RGB")

        # Get original dimensions
        width, height = img.size

        # Make the image square by padding the shorter dimension
        max_dim = max(width, height)

        # Calculate padding
        left = (max_dim - width) // 2
        top = (max_dim - height) // 2

        # Calculate scale factor for resizing
        scale = target_size / max_dim

        # Calculate final coordinates of original image in target space
        x1 = left * scale
        y1 = top * scale
        x2 = (left + width) * scale
        y2 = (top + height) * scale

        # Store original image coordinates and scale
        original_coords.append(np.array([x1, y1, x2, y2, width, height]))

        # Create a new black square image and paste original
        square_img = Image.new("RGB", (max_dim, max_dim), (0, 0, 0))
        square_img.paste(img, (left, top))

        # Resize to target size
        square_img = square_img.resize((target_size, target_size), Image.Resampling.BICUBIC)

        # Convert to tensor
        img_tensor = to_tensor(square_img)
        images.append(img_tensor)

    # Stack all images
    images = torch.stack(images)
    original_coords = torch.from_numpy(np.array(original_coords)).float()

    # Add additional dimension if single image to ensure correct shape
    if len(image_path_list) == 1:
        if images.dim() == 3:
            images = images.unsqueeze(0)
            original_coords = original_coords.unsqueeze(0)

    return images, original_coords


def load_and_preprocess_images(data_dir, mode="crop", start_frame=None, end_frame=None):
    """
    Loads and preprocesses a sequence of multi-view images from a directory.

    Directory Structure:
        data_dir/
            ├── frame_0/ (contains view images, e.g., CAM_F.jpg, CAM_B.jpg...)
            ├── frame_1/
            ...

    Args:
        data_dir (str): Root directory containing 'frame_x' folders.
        mode (str, optional): "crop" (default) or "pad".
        start_frame (int, optional): Index of the start frame (inclusive). Default is 0.
        end_frame (int, optional): Index of the end frame (inclusive). Default is all frames.

    Returns:
        torch.Tensor: Batched tensor with shape (B, T, V, 3, H, W).
                      B=1 (batch), T=frames, V=views per frame.
    """
    
    if not os.path.exists(data_dir):
        raise ValueError(f"Directory does not exist: {data_dir}")

    # 1. Parse frame directories
    frame_dirs = [d for d in os.listdir(data_dir) if os.path.isdir(os.path.join(data_dir, d)) and 'frame_' in d]
    
    if len(frame_dirs) == 0:
        raise ValueError(f"No 'frame_x' directories found in {data_dir}")

    # Sort frames numerically (e.g., frame_2 comes before frame_10)
    try:
        frame_dirs.sort(key=lambda x: int(x.split('_')[-1]))
    except ValueError:
        print("Warning: Could not sort frames numerically, falling back to lexicographical sort.")
        frame_dirs.sort()

    # 2. Slice frames based on start/end arguments
    total_frames = int(frame_dirs[-1].split('_')[-1])
    start_idx = max(0, start_frame) if start_frame is not None else 0
    end_idx = min(total_frames, end_frame) if end_frame is not None else total_frames
    selected_frames = [x for x in frame_dirs if start_idx <= int(x.split('_')[-1]) <= end_idx]
    
    if len(selected_frames) == 0:
        raise ValueError("No frames selected based on start_frame/end_frame range.")

    # 3. Collect image paths
    image_path_list = []
    num_views_per_frame = None
    
    for f_dir_name in selected_frames:
        f_path = os.path.join(data_dir, f_dir_name)
        valid_exts = ('.jpg', '.jpeg', '.png', '.bmp', '.tiff')
        img_files = [f for f in os.listdir(f_path) if f.lower().endswith(valid_exts)]
        
        # Sort views alphabetically to ensure consistent order (e.g., CAM_B0, CAM_F0...)
        img_files.sort()
        
        current_views = len(img_files)
        if current_views == 0:
            continue
            
        # Ensure every frame has the same number of views
        if num_views_per_frame is None:
            num_views_per_frame = current_views
        elif current_views != num_views_per_frame:
            raise ValueError(f"Inconsistent number of views in {f_dir_name}. Expected {num_views_per_frame}, got {current_views}.")
            
        for img_file in img_files:
            image_path_list.append(os.path.join(f_path, img_file))

    # T: Time steps (frames), V: Views per frame
    T = len(selected_frames)
    V = num_views_per_frame
    
    # --- Image Processing Logic ---

    if mode not in ["crop", "pad"]:
        raise ValueError("Mode must be either 'crop' or 'pad'")

    images = []
    shapes = set()
    to_tensor = TF.ToTensor()
    target_size = 512

    for image_path in image_path_list:
        img = Image.open(image_path)

        # Handle alpha channel
        if img.mode == "RGBA":
            background = Image.new("RGBA", img.size, (255, 255, 255, 255))
            img = Image.alpha_composite(background, img)

        img = img.convert("RGB")
        width, height = img.size

        # Calculate new dimensions
        if mode == "pad":
            if width >= height:
                new_width = target_size
                new_height = round(height * (new_width / width) / 16) * 16
            else:
                new_height = target_size
                new_width = round(width * (new_height / height) / 16) * 16
        else:  # mode == "crop"
            new_width = target_size
            new_height = round(height * (new_width / width) / 16) * 16

        img = img.resize((new_width, new_height), Image.Resampling.BICUBIC)
        img = to_tensor(img)

        # Center crop height if needed (crop mode)
        if mode == "crop" and new_height > target_size:
            start_y = (new_height - target_size) // 2
            img = img[:, start_y : start_y + target_size, :]

        # Pad to square (pad mode)
        if mode == "pad":
            h_padding = target_size - img.shape[1]
            w_padding = target_size - img.shape[2]

            if h_padding > 0 or w_padding > 0:
                pad_top = h_padding // 2
                pad_bottom = h_padding - pad_top
                pad_left = w_padding // 2
                pad_right = w_padding - pad_left

                img = torch.nn.functional.pad(
                    img, (pad_left, pad_right, pad_top, pad_bottom), mode="constant", value=1.0
                )

        shapes.add((img.shape[1], img.shape[2]))
        images.append(img)

    # Handle variable shapes across images
    if len(shapes) > 1:
        max_height = max(shape[0] for shape in shapes)
        max_width = max(shape[1] for shape in shapes)

        padded_images = []
        for img in images:
            h_padding = max_height - img.shape[1]
            w_padding = max_width - img.shape[2]

            if h_padding > 0 or w_padding > 0:
                pad_top = h_padding // 2
                pad_bottom = h_padding - pad_top
                pad_left = w_padding // 2
                pad_right = w_padding - pad_left

                img = torch.nn.functional.pad(
                    img, (pad_left, pad_right, pad_top, pad_bottom), mode="constant", value=1.0
                )
            padded_images.append(img)
        images = padded_images

    # 4. Final Stack and Reshape
    images = torch.stack(images)  # Shape: (N, C, H, W), where N = T * V
    
    N, C, H, W = images.shape
    assert N == T * V, f"Shape mismatch: Tensor N={N} != T={T} * V={V}"

    # Reshape to (B, T, V, C, H, W) with B=1
    images = images.view(1, T, V, C, H, W)

    return images