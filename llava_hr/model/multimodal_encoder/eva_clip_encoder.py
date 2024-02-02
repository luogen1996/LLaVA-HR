import torch
import torch.nn as nn
from timm import create_model
from transformers import CLIPImageProcessor
from torch.utils.checkpoint import checkpoint



cfg={
    "crop_size": 224,
    "do_center_crop": True,
    "do_normalize": True,
    "do_resize": True,
    "feature_extractor_type": "CLIPFeatureExtractor",
    "image_mean": [
        0.48145466,
        0.4578275,
        0.40821073
    ],
    "image_std": [
        0.26862954,
        0.26130258,
        0.27577711
    ],
    "resample": 3,
    "size": 224
}

class EVACLIPVisionTower(nn.Module):
    def __init__(self, vision_tower, args, delay_load=False):
        super().__init__()

        self.is_loaded = False
        self.freeze_vision=args.freeze_vision
        self.input_image_size=args.input_image_size
        self.vision_tower_name = vision_tower
        self.select_layer = args.mm_vision_select_layer
        self.select_feature = getattr(args, 'mm_vision_select_feature', 'patch')

        self.load_model()

    def load_model(self):
        self.image_processor = CLIPImageProcessor(**cfg)
        self.vision_tower = create_model(self.vision_tower_name,pretrained=True,dynamic_img_size=True)
        if self.freeze_vision:
            self.vision_tower.requires_grad_(False)

        if self.input_image_size is not None:
            self.image_processor.size=self.input_image_size
            self.image_processor.crop_size={
                'height':self.input_image_size,
                'width': self.input_image_size
            }
        setattr(self.vision_tower,'hidden_size',self.vision_tower.embed_dim)
        self.is_loaded = True

    def feature_select(self, image_forward_outs):

        if self.select_layer>100:
            n_layer=len(image_forward_outs)
            mod_index = max(n_layer // (self.select_layer//100),1)
            image_features=[]
            for i in range(n_layer):
                if  (i+1)% mod_index==0:
                    image_features.append(image_forward_outs[i])
            image_features=torch.cat(image_features,-1)
        else:
            image_features = image_forward_outs[self.select_layer]
        if self.select_feature == 'patch':
            image_features = image_features[:, 1:]
        elif self.select_feature == 'cls_patch':
            image_features = image_features
        else:
            raise ValueError(f'Unexpected select feature: {self.select_feature}')
        return image_features

    def forward_features(self, x):
        x = self.vision_tower.patch_embed(x)
        x, rot_pos_embed = self.vision_tower._pos_embed(x)
        image_forward_out=[]
        for blk in self.vision_tower.blocks:
            if self.vision_tower.grad_checkpointing and not torch.jit.is_scripting():
                x = checkpoint(blk, x, rope=rot_pos_embed)
            else:
                x = blk(x, rope=rot_pos_embed)
            image_forward_out.append(x)
        return image_forward_out

    def forward(self, images):
        if self.freeze_vision:
            with torch.no_grad():
                image_features = self._forward_images(images)
        else:
            image_features = self._forward_images(images)

        return image_features

    def _forward_images(self, images):
        if type(images) is list:
            image_features = []
            for image in images:
                image_forward_out = self.forward_features(image.to(device=self.device, dtype=self.dtype).unsqueeze(0))
                image_feature = self.feature_select(image_forward_out).to(image.dtype)
                image_features.append(image_feature)
        else:
            image_forward_outs = self.forward_features(images.to(device=self.device, dtype=self.dtype))
            image_features = self.feature_select(image_forward_outs).to(images.dtype)

        return image_features

    @property
    def dummy_feature(self):
        return torch.zeros(1, self.hidden_size, device=self.device, dtype=self.dtype)

    @property
    def dtype(self):
        return next(self.vision_tower.parameters()).dtype

    @property
    def device(self):
        return next(self.vision_tower.parameters()).device

    @property
    def config(self):
        assert  NotImplementedError
        pass

    @property
    def num_attention_heads(self):
        # as constant
        return 16
    @property
    def num_layers(self):
        # as constant
        return 24
    @property
    def hidden_size(self):
        return self.vision_tower.hidden_size

    @property
    def num_patches(self):
        return (cfg['image_size'] // self.patch_embed.patch_size[0]) ** 2
