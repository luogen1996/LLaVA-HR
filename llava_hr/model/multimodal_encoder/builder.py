import os
from .clip_encoder import CLIPVisionTower
from .eva_clip_encoder import EVACLIPVisionTower
from .multipath_encoder_wapper import MultiPathCLIPVisionTower
from .convnext_encoder import ConvNextVisionTower


def build_vision_tower(vision_tower_cfg, **kwargs):
    vision_tower = getattr(vision_tower_cfg, 'mm_vision_tower', getattr(vision_tower_cfg, 'vision_tower', None))
    is_absolute_path_exists = os.path.exists(vision_tower)
    is_multipath_encoder=getattr(vision_tower_cfg, 'is_multipath_encoder')
    if is_multipath_encoder:
        return MultiPathCLIPVisionTower(vision_tower, args=vision_tower_cfg, **kwargs)
    else:
        if is_absolute_path_exists or vision_tower.startswith("openai") or vision_tower.startswith("laion"):
            return CLIPVisionTower(vision_tower, args=vision_tower_cfg, **kwargs)
        elif vision_tower.startswith("eva"):
            return EVACLIPVisionTower(vision_tower, args=vision_tower_cfg, **kwargs)
        elif 'convnext' in vision_tower:
            return ConvNextVisionTower(vision_tower, args=vision_tower_cfg, **kwargs)

    raise ValueError(f'Unknown vision tower: {vision_tower}')
