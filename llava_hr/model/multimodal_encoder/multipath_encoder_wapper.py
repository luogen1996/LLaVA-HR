import torch
import torch.nn as nn
from torch.utils.checkpoint import checkpoint
from .convnext_encoder import ConvNextVisionTower
from .clip_encoder import CLIPVisionTower
import torch.nn.functional as F
from torch.nn.init import trunc_normal_
from copy import deepcopy
import random
import math


class MultiPathAlignModule(nn.Module):
    def __init__(self, fast_vision_dim, slow_vision_dim):
        super().__init__()

        self.fast_proj = nn.Linear(fast_vision_dim, fast_vision_dim)
        self.slow_proj = nn.Linear(slow_vision_dim, fast_vision_dim)

    def forward(self, fast_feat, slow_feat):
        if slow_feat.ndim == 4:
            b, c, h, w = slow_feat.shape
            slow_feat = slow_feat.view(b, c, -1).transpose(1, 2)
        assert slow_feat.shape[1] % fast_feat.shape[1] == 0 or fast_feat.shape[1] % slow_feat.shape[1] == 0
        if slow_feat.shape[1] < fast_feat.shape[1]:
            # upsample
            b, l, c = slow_feat.shape
            src_size = int(math.sqrt(l))
            dst_size = int(math.sqrt(fast_feat.shape[1]))
            slow_feat = slow_feat.transpose(1, 2).view(b, c, src_size, src_size)
            slow_feat = F.interpolate(slow_feat.float(), size=(dst_size, dst_size), mode='bilinear',
                                      align_corners=True).to(dtype=slow_feat.dtype)
            slow_feat = slow_feat.view(b, c, -1).transpose(1, 2)
        elif slow_feat.shape[1] > fast_feat.shape[1]:
            # pooling
            b, l, c = slow_feat.shape
            src_size = int(math.sqrt(l))
            dst_size = int(math.sqrt(fast_feat.shape[1]))
            slow_feat = slow_feat.transpose(1, 2).view(b, c, src_size, src_size)
            slow_feat = F.avg_pool2d(slow_feat, src_size // dst_size, src_size // dst_size)
            slow_feat = slow_feat.view(b, c, -1).transpose(1, 2)
        patch_feat = self.fast_proj(fast_feat) + self.slow_proj(slow_feat)
        return patch_feat


class S2FStitchAlignModuleV2(nn.Module):
    def __init__(self, fast_vision_dim, slow_vision_dim, zero_init=True):
        super().__init__()

        self.slow_conv = nn.Conv2d(slow_vision_dim, slow_vision_dim, 1)
        self.slow_proj = nn.Conv2d(slow_vision_dim, fast_vision_dim, 1)

        self.fast_conv = nn.Conv2d(fast_vision_dim, fast_vision_dim, 7, padding=3, groups=fast_vision_dim)
        self.fast_proj = nn.Conv2d(fast_vision_dim, fast_vision_dim, 1)

        self.gate = nn.Sequential(
            nn.Linear(fast_vision_dim*2, fast_vision_dim//2),
            nn.GELU(),
            nn.Linear(fast_vision_dim//2, 1) )

        nn.init.xavier_uniform_(self.slow_conv.weight)
        nn.init.xavier_uniform_(self.fast_conv.weight)
        nn.init.zeros_(self.slow_conv.bias)
        nn.init.zeros_(self.fast_conv.bias)
        if zero_init:
            nn.init.zeros_(self.slow_proj.weight)
            nn.init.zeros_(self.fast_proj.weight)
        else:
            nn.init.xavier_uniform_(self.slow_proj.weight)
            nn.init.xavier_uniform_(self.fast_proj.weight)
        nn.init.zeros_(self.slow_proj.bias)
        nn.init.zeros_(self.fast_proj.bias)

    def src2dst_align(self, src_feat, dst_feat):
        dst_size = int(math.sqrt(dst_feat.shape[1]))
        assert src_feat.shape[1] % dst_feat.shape[1] == 0 or dst_feat.shape[1] % src_feat.shape[1] == 0
        if src_feat.shape[1] < dst_feat.shape[1]:
            # upsample
            b, l, c = src_feat.shape
            src_size = int(math.sqrt(l))
            dst_size = int(math.sqrt(dst_feat.shape[1]))
            src_feat = src_feat.transpose(1, 2).view(b, c, src_size, src_size)
            src_feat = F.interpolate(src_feat.float(), size=(dst_size, dst_size), mode='bilinear',
                                     align_corners=True).to(dtype=src_feat.dtype)
            src_feat = src_feat.view(b, c, -1).transpose(1, 2)
        elif src_feat.shape[1] > dst_feat.shape[1]:
            # pooling
            b, l, c = src_feat.shape
            src_size = int(math.sqrt(l))
            dst_size = int(math.sqrt(dst_feat.shape[1]))
            src_feat = src_feat.transpose(1, 2).view(b, c, src_size, src_size)
            src_feat = F.avg_pool2d(src_feat, src_size // dst_size, src_size // dst_size)
            src_feat = src_feat.view(b, c, -1).transpose(1, 2)
        return src_feat, dst_size

    def forward(self, fast_feat, slow_feat):
        b, c, h, w = slow_feat.shape
        _, _, d = fast_feat.shape
        slow_feat = self.slow_proj(F.gelu(self.slow_conv(slow_feat)))
        slow_feat = slow_feat.view(b, d, -1).transpose(1, 2)
        slow_feat_align, dst_size = self.src2dst_align(slow_feat, fast_feat)
        fast_feat = fast_feat.transpose(1, 2).view(b, d, dst_size, dst_size)
        fast_feat = fast_feat + self.fast_proj(F.gelu(self.fast_conv(fast_feat)))
        fast_feat = fast_feat.view(b, d, dst_size * dst_size).transpose(1, 2)
        gate=self.gate(torch.cat([fast_feat,slow_feat_align],-1).mean(1)).unsqueeze(1)
        fast_feat = fast_feat + slow_feat_align *gate.tanh()
        return fast_feat


class MultiPathCLIPVisionTower(nn.Module):
    def __init__(self, vision_tower, args, delay_load=False):
        super().__init__()

        self.is_loaded = False

        self.slow_vision_tower = ConvNextVisionTower(args.vision_tower_slow, args,
                                                     delay_load=delay_load)
        args_ = deepcopy(args)
        # set a default image size
        args_.input_image_size = 336
        self.fast_vision_tower = CLIPVisionTower(vision_tower, args_, delay_load=delay_load)

        self.load_model()

        self.vision_tower_name = vision_tower
        self.select_layer = args.mm_vision_select_layer
        self.select_feature = getattr(args, 'mm_vision_select_feature', 'patch')
        self.splits = self.select_layer // 100 if self.select_layer > 100 else 1
        self.enable_adapter= not args.freeze_vision
        self.image_size=args.input_image_size


        if self.enable_adapter:
            self.align_stages_latent = nn.ModuleList([S2FStitchAlignModuleV2(self.fast_vision_tower.hidden_size,
                                                                             self.slow_vision_tower.hidden_size,
                                                                             True)
                                                      for i in range(3)])

        self.align_stages = nn.ModuleList([MultiPathAlignModule(self.fast_vision_tower.hidden_size,
                                                                self.slow_vision_tower.hidden_size
                                                                )
                                           ])

    def load_model(self):
        self.slow_vision_tower.load_model()
        self.fast_vision_tower.load_model()
        self.image_processor = self.slow_vision_tower.image_processor

        self.is_loaded = True

    def forward(self, x):

        # fast & slow brach
        fast_blk = self.fast_vision_tower.vision_tower.vision_model.encoder.layers
        slow_blk = self.slow_vision_tower.vision_tower.stages
        n_blks = len(fast_blk) // 4
        assert len(fast_blk) == n_blks * 4

        # pre-process for fast_vision_towe
        fast_image_size=max(int(self.image_size/32*14),336)
        y = F.interpolate(x.float(), size=(fast_image_size, fast_image_size), mode='bilinear', align_corners=True).to(dtype=x.dtype)
        y = self.fast_vision_tower.vision_tower.vision_model.embeddings(y)
        y = self.fast_vision_tower.vision_tower.vision_model.pre_layrnorm(y[:, 1:])

        # pre-process for slow_vision_tower
        x = self.slow_vision_tower.vision_tower.stem(x)

        #inference of slow branch
        x = slow_blk[0](x)
        x = slow_blk[1](x)
        x = slow_blk[2](x)
        x = slow_blk[3](x)

        # inference of fast branch
        for blk in fast_blk[:n_blks]:
            if self.training:
                y=checkpoint(
                    blk.__call__,
                    y,
                    None,
                    None
                )[0]
            else:
                y = blk(y, None, None)[0]
        if self.enable_adapter:
            y = self.align_stages_latent[0](y, x)

        for blk in fast_blk[n_blks:2 * n_blks]:
            if self.training:
                y=checkpoint(
                    blk.__call__,
                    y,
                    None,
                    None
                )[0]
            else:
                y = blk(y, None, None)[0]

        if self.enable_adapter:
            y = self.align_stages_latent[1](y, x)
        for blk in fast_blk[2 * n_blks:3 * n_blks]:
            if self.training:
                y=checkpoint(
                    blk.__call__,
                    y,
                    None,
                    None
                )[0]
            else:
                y = blk(y, None, None)[0]
        if self.enable_adapter:
            y = self.align_stages_latent[2](y, x)
        for blk in fast_blk[3 * n_blks:]:
            if self.training:
                y=checkpoint(
                    blk.__call__,
                    y,
                    None,
                    None
                )[0]
            else:
                y = blk(y, None, None)[0]

        #features combination
        y = self.align_stages[0](y, x)

        return y

    def forward_features(self, x):
        assert  NotImplementedError

    @property
    def dummy_feature(self):
        return torch.zeros(1, self.hidden_size, device=self.device, dtype=self.dtype)

    @property
    def dtype(self):
        return next(self.fast_vision_tower.parameters()).dtype

    @property
    def device(self):
        return next(self.fast_vision_tower.parameters()).device

    @property
    def config(self):
        assert NotImplementedError
        pass

    @property
    def hidden_size(self):
        return self.fast_vision_tower.hidden_size

    @property
    def num_patches(self):
        return self.fast_vision_tower.num_patches
