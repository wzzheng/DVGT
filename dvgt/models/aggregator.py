import copy
import torch
import torch.nn as nn
from torch.utils.checkpoint import checkpoint
from typing import Tuple, List

from dvgt.layers.block import Block
from dvgt.layers.rope import RotaryPositionEmbedding2D, PositionGetter
from dvgt.layers.sinusoid_pose_encoding import get_sinusoid_encoding_table

_RESNET_MEAN = [0.485, 0.456, 0.406]
_RESNET_STD = [0.229, 0.224, 0.225]


class Aggregator(nn.Module):
    """
    The Aggregator applies alternating-attention over input frames,
    as described in DVGT: Visual Geometry Transformer for autonomous Driving.

    Remember to set model.train() to enable gradient checkpointing to reduce memory usage.

    Args:
        patch_size (int): Patch size for image backbone.
        embed_dim (int): Dimension of the token embeddings.
        depth (int): Number of blocks.
        num_heads (int): Number of attention heads.
        mlp_ratio (float): Ratio of MLP hidden dim to embedding dim.
        num_register_tokens (int): Number of register tokens.
        block_fn (nn.Module): The block type used for attention (Block by default).
        qkv_bias (bool): Whether to include bias in QKV projections.
        proj_bias (bool): Whether to include bias in the output projection.
        ffn_bias (bool): Whether to include bias in MLP layers.
        patch_embed (str): Type of patch embed. e.g., "dinov3_vitl16".
        dino_v3_weight_path (str): Weight path for dinov3 corresponding to the patch embed.
        aa_order (list[str]): The order of alternating attention, e.g. ["intra_view", "cross_view", "cross_frame"].
        aa_block_size (int): How many blocks to group under each attention type before switching. If not necessary, set to 1.
        qk_norm (bool): Whether to apply QK normalization.
        rope_freq (int): Base frequency for rotary embedding. -1 to disable.
        init_values (float): Init scale for layer scale.
        max_frames_for_temporal_pos_embed (int): Maximum number of frames for temporal positional embedding.
    """

    def __init__(
        self,
        patch_size=16,
        embed_dim=1024,
        depth=24,
        num_heads=16,
        mlp_ratio=4.0,
        num_register_tokens=4,
        block_fn=Block,
        qkv_bias=True,
        proj_bias=True,
        ffn_bias=True,
        patch_embed="dinov3_vitl16",
        dino_v3_weight_path="ckpt/dinov3_vitl16_pretrain_lvd1689m-8aa4cbdd.pth",
        aa_order=["intra_view", "cross_view", "cross_frame"],
        aa_block_size=1,
        qk_norm=True,
        rope_freq=100,
        init_values=0.01,
        max_frames_for_temporal_pos_embed=24,
    ):
        super().__init__()

        # self.patch_embed = torch.hub.load('./dinov3', patch_embed, source='local', weights=dino_v3_weight_path)   # Load model code and weight
        self.patch_embed = torch.hub.load('./dinov3', patch_embed, source='local')      # Only load model code
        # Disable gradient updates for mask token
        if hasattr(self.patch_embed, "mask_token"):
            self.patch_embed.mask_token.requires_grad_(False)

        # Initialize rotary position embedding if frequency > 0
        self.rope = RotaryPositionEmbedding2D(frequency=rope_freq) if rope_freq > 0 else None
        self.position_getter = PositionGetter() if self.rope is not None else None
        self.aa_order = aa_order
        self.embed_dim = embed_dim

        self.intra_view_blocks = nn.ModuleList(
            [
                block_fn(
                    dim=embed_dim,
                    num_heads=num_heads,
                    mlp_ratio=mlp_ratio,
                    qkv_bias=qkv_bias,
                    proj_bias=proj_bias,
                    ffn_bias=ffn_bias,
                    init_values=init_values,
                    qk_norm=qk_norm,
                    rope=self.rope,
                )
                for _ in range(depth)
            ]
        )

        self.cross_view_blocks = copy.deepcopy(self.intra_view_blocks)
    
        self.cross_frame_blocks = copy.deepcopy(self.intra_view_blocks)

        self.depth = depth
        self.patch_size = patch_size
        assert aa_block_size == 1, "aa_block_size must be 1 for now"
        self.aa_block_size = aa_block_size

        # Validate that depth is divisible by aa_block_size
        if self.depth % self.aa_block_size != 0:
            raise ValueError(f"depth ({depth}) must be divisible by aa_block_size ({aa_block_size})")

        self.aa_block_num = self.depth // self.aa_block_size

        self.patch_start_idx = num_register_tokens + 1
        # The patch tokens start after the ego and register tokens
        self.register_token = nn.Parameter(torch.randn(1, 2, num_register_tokens, embed_dim))
        nn.init.normal_(self.register_token, std=1e-6)

        # Note: We have two ego pose tokens, the one for first frame and all views, the other for other frame and all views. 
        # The same applies for register tokens
        self.ego_pose_token = nn.Parameter(torch.randn(1, 2, 1, embed_dim))
        nn.init.normal_(self.ego_pose_token, std=1e-6)

        # Register normalization constants as buffers
        for name, value in (("_resnet_mean", _RESNET_MEAN), ("_resnet_std", _RESNET_STD)):
            self.register_buffer(name, torch.FloatTensor(value).view(1, 1, 1, 3, 1, 1), persistent=False)

        self.temporal_pos_embed = nn.Parameter(torch.randn(1, max_frames_for_temporal_pos_embed, embed_dim))
        nn.init.normal_(self.temporal_pos_embed, std=1e-6)

        self.use_reentrant = False # hardcoded to False

    def forward(self, images: torch.Tensor) -> Tuple[List[torch.Tensor], int]:
        """
        Args:
            images (torch.Tensor): Input images with shape [B, T, V, 3, H, W], in range [0, 1].
                B: batch size, T: num_frames, V: views_per_frame, 3: RGB channels, H: height, W: width

        Returns:
            (list[torch.Tensor], int):
                The list of outputs from the attention blocks,
                and the patch_start_idx indicating where patch tokens begin.
        """
        B, T, V, C_in, H, W = images.shape

        if C_in != 3:
            raise ValueError(f"Expected 3 input channels, got {C_in}")

        # Normalize images and reshape for patch embed
        images = (images - self._resnet_mean) / self._resnet_std

        # Reshape to [B*T*V, C, H, W] for patch embedding
        images = images.view(B * T * V, C_in, H, W)
        patch_tokens = self.patch_embed.forward_features(images)

        if isinstance(patch_tokens, dict):
            patch_tokens = patch_tokens["x_norm_patchtokens"]

        _, P, C = patch_tokens.shape

        patch_tokens = patch_tokens.view(B, T, V, P, C)

        if T > self.temporal_pos_embed.shape[1]:
            raise ValueError(f"Input frames T={T} exceeds max_frames_for_temporal_pos_embed={self.temporal_pos_embed.shape[1]} for temporal embedding.")
        # self.temporal_pos_embed -> [1, T, C], reshape for broadcasting -> [1, T, 1, 1, C]
        temporal_embed = self.temporal_pos_embed[:, :T, :][:, :, None, None, :].contiguous()
        patch_tokens = patch_tokens + temporal_embed
    
        # Reshape back to [B*T*V, P, C] for subsequent processing
        patch_tokens = patch_tokens.view(B * T * V, P, C)

        # Expand ego and register tokens to match batch size and sequence length
        register_token = slice_expand_and_flatten(self.register_token, B, T, V)
        ego_pose_token = slice_expand_and_flatten(self.ego_pose_token, B, T, V)

        # Concatenate special tokens with patch tokens
        tokens = [ego_pose_token, register_token, patch_tokens]
        tokens = torch.cat(tokens, dim=1)

        pos = None
        if self.rope is not None:
            pos = self.position_getter(B * T * V, H // self.patch_size, W // self.patch_size, device=images.device)

        if self.patch_start_idx > 0:
            # do not use position embedding for special tokens (camera and register tokens)
            # so set pos to 0 for the special tokens
            pos = pos + 1
            pos_special = torch.zeros(B * T * V, self.patch_start_idx, 2).to(images.device).to(pos.dtype)
            pos = torch.cat([pos_special, pos], dim=1)

        # update P because we added special tokens
        _, P, C = tokens.shape
        intra_view_idx, cross_view_idx, cross_frame_idx = 0, 0, 0
        output_list = []

        for i in range(self.aa_block_num):
            for attn_type in self.aa_order:
                if attn_type == "intra_view":
                    tokens, intra_view_idx, intra_view_intermediates = self._process_intra_view_attention(
                        tokens, B, T, V, P, C, intra_view_idx, pos=pos
                    )
                elif attn_type == "cross_view":
                    tokens, cross_view_idx, cross_view_intermediates = self._process_cross_view_attention(
                        tokens, B, T, V, P, C, cross_view_idx, pos=pos
                    )
                elif attn_type == "cross_frame":
                    tokens, cross_frame_idx, cross_frame_intermediates = self._process_cross_frame_attention(
                        tokens, B, T, V, P, C, cross_frame_idx, pos=pos
                    )
                else:
                    raise ValueError(f"Unknown attention type: {attn_type}")

            for j in range(len(intra_view_intermediates)):
                # concat intermediates, [B*T*V, P, 3C]
                concat_inter = [cross_frame_intermediates[j], cross_view_intermediates[j], intra_view_intermediates[j]]
                output_list.append(torch.cat(concat_inter, dim=-1))

        del concat_inter, intra_view_intermediates, cross_view_intermediates, cross_frame_intermediates

        return output_list, self.patch_start_idx

    def _process_intra_view_attention(self, tokens, B, T, V, P, C, intra_view_idx, pos=None):
        """
        Process intra_view attention blocks. We keep tokens in shape (B*T*V, P, C).
        """
        if tokens.shape != (B * T * V, P, C):
            tokens = tokens.view(B, T, V, P, C).view(B * T * V, P, C)

        if pos is not None and pos.shape != (B * T * V, P, 2):
            pos = pos.view(B, T, V, P, 2).view(B * T * V, P, 2)

        intermediates = []

        # by default, self.aa_block_size=1, which processes one block at a time
        for _ in range(self.aa_block_size):
            if self.training:
                tokens = checkpoint(self.intra_view_blocks[intra_view_idx], tokens, pos, use_reentrant=self.use_reentrant)
            else:
                tokens = self.intra_view_blocks[intra_view_idx](tokens, pos=pos)
            intra_view_idx += 1
            intermediates.append(tokens.view(B, T, V, P, C))

        return tokens, intra_view_idx, intermediates

    def _process_cross_view_attention(self, tokens, B, T, V, P, C, cross_view_idx, pos=None):
        """
        Process cross_view attention blocks. We keep tokens in shape (B*T, V*P, C).
        """
        # If needed, reshape tokens or positions:
        if tokens.shape != (B * T, V * P, C):
            tokens = tokens.view(B, T, V, P, C).view(B * T, V * P, C)

        if pos is not None and pos.shape != (B * T, V * P, 2):
            pos = pos.view(B, T, V, P, 2).view(B * T, V * P, 2)

        intermediates = []

        # by default, self.aa_block_size=1, which processes one block at a time
        for _ in range(self.aa_block_size):
            if self.training:
                tokens = checkpoint(self.cross_view_blocks[cross_view_idx], tokens, pos, use_reentrant=self.use_reentrant)
            else:
                tokens = self.cross_view_blocks[cross_view_idx](tokens, pos=pos)
            cross_view_idx += 1
            intermediates.append(tokens.view(B, T, V, P, C))

        return tokens, cross_view_idx, intermediates

    def _process_cross_frame_attention(self, tokens, B, T, V, P, C, cross_frame_idx, pos=None):
        """
        Process cross_frame attention blocks. We keep tokens in shape (B*V, T*P, C).
        """
        tokens = tokens.view(B, T, V, P, C).transpose(1, 2).contiguous().view(B * V, T * P, C)

        if pos is not None:
            pos = pos.view(B, T, V, P, 2).transpose(1, 2).contiguous().view(B * V, T * P, 2)

        intermediates = []

        # by default, self.aa_block_size=1, which processes one block at a time
        for _ in range(self.aa_block_size):
            if self.training:
                tokens = checkpoint(self.cross_frame_blocks[cross_frame_idx], tokens, pos, use_reentrant=self.use_reentrant)
            else:
                tokens = self.cross_frame_blocks[cross_frame_idx](tokens, pos=pos)
            cross_frame_idx += 1
            tokens = tokens.view(B, V, T, P, C).transpose(1, 2).contiguous()
            intermediates.append(tokens)

        return tokens, cross_frame_idx, intermediates

def slice_expand_and_flatten(token_tensor, B, T, V):
    """
    Processes specialized tokens with shape (1, 2, X, C) for multi-frame processing:
    1) Uses the first position (index=0) for the first 1*V frame only
    2) Uses the second position (index=1) for all remaining frames (T-1)*V frames
    3) Expands both to match batch size B, T
    4) Concatenates to form (B, T*V, X, C) where each sequence has 1 first-position token
       followed by (T*V-1) second-position tokens
    5) Flattens to (B*T*V, X, C) for processing

    Returns:
        torch.Tensor: Processed tokens with shape (B*T*V, X, C)
    """

    # Slice out the "query" tokens => shape (B, V, ...)
    query = token_tensor[:, 0:1, ...].expand(B, V, *token_tensor.shape[2:])
    # Slice out the "other" tokens => shape (B, (T - 1) * V, ...)
    others = token_tensor[:, 1:, ...].expand(B, (T - 1) * V, *token_tensor.shape[2:])
    # Concatenate => shape (B, T*V, ...)
    combined = torch.cat([query, others], dim=1)

    # Finally flatten => shape (B*T*V, ...)
    combined = combined.view(B * T * V, *combined.shape[2:])
    return combined
