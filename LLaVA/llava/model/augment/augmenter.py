import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from torchvision.transforms.functional import gaussian_blur
from torchvision import utils
from typing import Tuple, Optional


class AugmentedImageProcess(nn.Module):
    def __init__(self,
                 aug_type: str = "none",
                 merge_num: int = 0,
                 lam: Optional[float] = None,
                 vision_patch: int = 14,  # For CLIP with 336 resolution
                 use_attn: bool = False,
                ):
        super().__init__()
        assert aug_type is not None, "aug_type cannot be None"
        self.aug_type = aug_type
        self.merge_num = merge_num
        self.lam = float(lam) if lam is not None else np.random.uniform(0.0, 1.0)
        self.vp = vision_patch
        self.use_attn = use_attn
        self.scope = (0.1, 0.8)
        
        valid_types = ['none', 'blur', 'mixup', 'cutmix', 'resizemix', 'mergemix', 'mergemix-r']
        if aug_type == 'resizemix':
            self.lam = np.random.uniform(self.scope[0], self.scope[-1])
        assert self.aug_type in valid_types, \
            f"Invalid aug_type: {aug_type}. Must be one of {valid_types}"


    def rand_bbox(self, size, tao=None):
        """ generate random box by lam / tao"""
        W = size[2]
        H = size[3]
        if tao is None:
            tao = np.sqrt(1. - self.lam)
        cut_w = int(W * tao)
        cut_h = int(H * tao)

        # uniform
        cx = np.random.randint(W)
        cy = np.random.randint(H)

        bbx1 = np.clip(cx - cut_w // 2, 0, W)
        bby1 = np.clip(cy - cut_h // 2, 0, H)
        bbx2 = np.clip(cx + cut_w // 2, 0, W)
        bby2 = np.clip(cy + cut_h // 2, 0, H)

        return bbx1, bby1, bbx2, bby2


    def token_unmerge(self, keep_tokens: torch.Tensor, source: Optional[torch.Tensor] = None) -> torch.Tensor:
        """ recovery full tokens with the given source_matrix """
        if source is None:
            return keep_tokens
            
        batch_size, _ = keep_tokens.shape
        full_length = source.shape[-1]  # [B, keep_L, full_L]
        full_tokens = torch.zeros(batch_size, full_length, device=keep_tokens.device)
        
        mask = source == 1
        batch_indices = torch.arange(batch_size, device=keep_tokens.device).unsqueeze(1)
        keep_indices = mask.nonzero()[:, 1].view(batch_size, -1)
        full_indices = mask.nonzero()[:, 2].view(batch_size, -1)
        
        full_tokens[batch_indices, full_indices] = keep_tokens[batch_indices, keep_indices]
        return full_tokens


    def use_blur(self, img: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        """Applying Gussian Blur for Augmentation"""
        h = img.shape[2]
        kernel_size = max(2, int(h * self.lam))
        kernel_size = kernel_size if kernel_size % 2 == 1 else kernel_size + 1
        img_ = gaussian_blur(img, (kernel_size, kernel_size), sigma=(0.5, 10.0))
        return img * (1 - mask) + mask * img_


    def use_mixup(self, img: torch.Tensor, batch_size: int) -> torch.Tensor:
        """Applying MixUp for Augmentation"""
        index = torch.randperm(batch_size, device=img.device)
        return img * self.lam + (1 - self.lam) * img[index, :], index
        

    def use_cutmix(self, img: torch.Tensor, batch_size: int, w: int, h: int) -> torch.Tensor:
        """Applying CutMix for Augmentation"""
        index = torch.randperm(batch_size, device=img.device)
        bbx1, bbx2, bby1, bby2 = self.rand_bbox(img.size())
        img[:, :, bbx1:bbx2, bby1:bby2] = img[index, :, bbx1:bbx2, bby1:bby2]
        lam = 1 - ((bbx2 - bbx1) * (bby2 - bby1) / (w * h))
        return img, lam, index


    def use_resizemix(self, img: torch.Tensor, batch_size: int, w: int, h: int) -> torch.Tensor:
        """Applying ResizeMix for Augmentation"""
        index = torch.randperm(batch_size, device=img.device)
        x1, y1, x2, y2 = self.rand_bbox(img.size(), tao=self.lam)

        region_w = int(x2 - x1)
        region_h = int(y2 - y1)
        if region_w <= 0 or region_h <= 0:
            return img, 1.0, index
        
        img_resize = F.interpolate(
            img[index], (region_h, region_w), mode='nearest'
        )
        img[:, :, y1:y2, x1:x2] = img_resize
        lam = 1 - (region_w * region_h / (w * h))
        return img, lam, index


    def use_mergemix(self, img: torch.Tensor, mask: torch.Tensor, batch_size: int) -> torch.Tensor:
        """Applying MergeMix for Augmentation"""
        index = torch.randperm(batch_size, device=img.device)
        return img * (1 - mask) + mask * img[index, :], index


    def forward(self, 
                img: torch.Tensor, 
                vision_out: Optional[Tuple[torch.Tensor, ...] | torch.Tensor] = None
               ) -> torch.Tensor:
        if vision_out is not None:
            assert isinstance(vision_out, (tuple, torch.Tensor)), \
                "vision_out must be a tensor or tuple of tensors"
        
        aug_info = {
                "lam": self.lam,
                "index": None
            }
        b, c, h, w = img.shape
        attention: Optional[torch.Tensor] = None
        source: Optional[torch.Tensor] = None
        if isinstance(vision_out, tuple) and len(vision_out) == 2:
            attention, source = vision_out[0], vision_out[1]
        elif isinstance(vision_out, (tuple, torch.Tensor)) and len(vision_out) == b:
            # If you use ToMe with zero merge, which will cause only return (attention socre), instead of (attention score, source matrix)
            # FIXME return attention score: len(vision_out) = batch size, vision_out.shape = batch size, multi-heads, 577, 577
            attention = vision_out

        tokens_num = (h * w) // (self.vp * self.vp)
        # mixup don't need topk attention patch
        if self.aug_type in ["mixup", "cutmix", "resizemix"]:
            if self.aug_type == "mixup":
                img, aug_info["index"] = self.use_mixup(img, b)
            elif self.aug_type == "cutmix":
                img, aug_info["lam"], aug_info["index"] = self.use_cutmix(img, b, w, h)
            elif self.aug_type == "resizemix":
                img, aug_info["lam"], aug_info["index"] = self.use_resizemix(img, b, w, h)
            return img, aug_info

        if not self.use_attn:  # Randomly retain tokens
            num_zeros = int(tokens_num * self.lam)
            indices = torch.randperm(tokens_num, device=img.device)[:num_zeros].repeat(b, 1)
        else:                  # TopK-based retain tokens
            if attention is None:
                raise ValueError("attention must be provided when use_attn is True")
            attention = attention[:, :, 0, 1:]   # Excluding [CLS] token
            attention_sum = attention.sum(dim=1)
            num_zeros = int(attention_sum.shape[-1] * self.lam)
            if self.aug_type == "mergemix-r":  # MergeMix with Random Tokens
                indices = torch.randperm(attention_sum.shape[-1], device=img.device)[:num_zeros].repeat(b, 1)
            else:
                indices = attention_sum.topk(num_zeros, dim=1).indices

        mask = torch.ones(b, tokens_num, device=img.device)
        mask.scatter_(1, indices, 0.0)

        # Recover to correct resolution
        if self.aug_type in ["mergemix", "mergemix-r"] and source is not None:
            mask = self.token_unmerge(mask, source[:, 1:, 1:])


        # Upsamping
        mask = mask.view(b, 1, h // self.vp, w // self.vp)
        mask = F.interpolate(mask, scale_factor=self.vp, mode='nearest')

        if self.aug_type == "blur":
            img = self.use_blur(img, mask)
        elif self.aug_type in ["mergemix", "mergemix-r"]:
            img, aug_info["index"] = self.use_mergemix(img, mask, b)

        return img, aug_info