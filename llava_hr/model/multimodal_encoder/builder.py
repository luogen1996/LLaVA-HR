import os
from .clip_encoder import CLIPVisionTower
from .eva_clip_encoder import EVACLIPVisionTower
from .multipath_encoder_wapper import MultiPathCLIPVisionTower
from .convnext_encoder import ConvNextVisionTower
import torch

def build_vision_tower(vision_tower_cfg, **kwargs):
    vision_tower = getattr(vision_tower_cfg, 'mm_vision_tower', getattr(vision_tower_cfg, 'vision_tower', None))
    is_absolute_path_exists = os.path.exists(vision_tower)
    is_multipath_encoder=getattr(vision_tower_cfg, 'is_multipath_encoder')
    if is_multipath_encoder:
        model = MultiPathCLIPVisionTower(vision_tower, args=vision_tower_cfg, **kwargs)
        if hasattr(vision_tower_cfg, "vision_tower_checkpoint") and vision_tower_cfg.vision_tower_checkpoint is not None:
            # from deepspeed.utils import safe_get_full_model
            # full_model = safe_get_full_model(model)

            vision_tower_state_dict = torch.load(vision_tower_cfg.vision_tower_checkpoint)

            new_vision_tower_state_dict = {}

            # 遍历原始state_dict中的每个键值对
            for key, value in vision_tower_state_dict.items():
                # 如果键以 '.model' 开头，则去掉该前缀
                if key.startswith('model.vision_tower.'):
                    new_key = key.replace('model.vision_tower.', '', 1)  # 只替换第一个出现的'.model'
                    new_vision_tower_state_dict[new_key] = value
                # else:
                #     new_key = key
                
                # 将新的键和对应的值存入新的state_dict
                    
            # 然后加载新的state_dict到模型中
            # for name, param in new_vision_tower_state_dict.items():
            #     print(f"new ckpt: {name}: {param.shape}") 
                # 使用 GatheredParameters 确保所有参数可见
            from deepspeed import zero
            with zero.GatheredParameters(list(model.parameters())):
                msg = model.load_state_dict(new_vision_tower_state_dict, strict=False)
                print(msg)
            print("successfully load vision tower from", vision_tower_cfg.vision_tower_checkpoint)
        return model
    else:
        if is_absolute_path_exists or vision_tower.startswith("openai") or vision_tower.startswith("laion"):
            return CLIPVisionTower(vision_tower, args=vision_tower_cfg, **kwargs)
        elif vision_tower.startswith("eva"):
            return EVACLIPVisionTower(vision_tower, args=vision_tower_cfg, **kwargs)
        elif 'convnext' in vision_tower:
            return ConvNextVisionTower(vision_tower, args=vision_tower_cfg, **kwargs)

    raise ValueError(f'Unknown vision tower: {vision_tower}')
