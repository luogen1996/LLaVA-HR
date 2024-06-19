import torch
import torch.nn as nn
from timm import create_model
from transformers import CLIPImageProcessor
from .convnext import convnext_base_clip,convnext_large_mlp,convnext_xxlarge,checkpoint_filter_fn
from torch.utils.checkpoint import checkpoint
import deepspeed
from transformers.deepspeed import deepspeed_config, is_deepspeed_zero3_enabled

cfg={
    "crop_size": 256,
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
    "size": 256
}

class ConvNextVisionTower(nn.Module):
    def __init__(self, vision_tower, args, delay_load=False):
        super().__init__()

        self.is_loaded = False
        self.freeze_vision=args.freeze_vision
        self.input_image_size=args.input_image_size
        self.vision_tower_name = vision_tower
        self.select_layer = args.mm_vision_select_layer
        self.select_feature = getattr(args, 'mm_vision_select_feature', 'patch')
        self.select_feature = getattr(args, 'mm_vision_select_feature', 'patch')
        # 0401 fix for fine-tune backbone
        self.enable_pretrain = (not args.tune_mm_mlp_adapter) and  ('llava' not in args._name_or_path)

        self.load_model()

    def load_model(self):
        self.image_processor = CLIPImageProcessor(**cfg)

        if 'xxlarge' in self.vision_tower_name:
            if self.enable_pretrain:
                self.vision_tower = convnext_xxlarge(self.vision_tower_name)
            else:
                self.vision_tower = convnext_xxlarge()
                #need a configuration to write load_path
                # self.load_checkpoint(self.vision_tower,'./checkpoints/convnext_xxlarge/open_clip_pytorch_model.bin')
            setattr(self.vision_tower, 'hidden_size', 3072)
        else:
            if self.enable_pretrain:
                self.vision_tower = convnext_large_mlp(self.vision_tower_name)
            else:
                self.vision_tower = convnext_large_mlp()
                # need a configuration to write load_path
                # self.load_checkpoint(self.vision_tower, './checkpoints/convnext_large/open_clip_pytorch_model.bin')
            setattr(self.vision_tower, 'hidden_size', 1536)
        if self.freeze_vision:
            self.vision_tower.requires_grad_(False)

        # if self.vision_tower.grad_checkpointing:
        for s in self.vision_tower.stages:
            s.grad_checkpointing = True

        if self.input_image_size is not None:
            self.image_processor.size=self.input_image_size
            self.image_processor.crop_size={
                'height':self.input_image_size,
                'width': self.input_image_size
            }

        self.is_loaded = True

    # modified function from open_clip to support zero3 stage
    def load_checkpoint(self,model, checkpoint_path, strict=True):
        state_dict = torch.load(checkpoint_path)
        state_dict=checkpoint_filter_fn(model,state_dict)
        if is_deepspeed_zero3_enabled():

            error_msgs = []

            def load(module: nn.Module, state_dict, prefix=""):
                metadata = None

                local_metadata = {} if metadata is None else metadata.get(prefix[:-1], {})
                args = (state_dict, prefix, local_metadata, True, [], [], error_msgs)
                # Parameters of module and children will start with prefix. We can exit early if there are none in this
                # state_dict
                if len([key for key in state_dict if key.startswith(prefix)]) > 0:
                    if is_deepspeed_zero3_enabled():
                        # In sharded models, each shard has only part of the full state_dict, so only gather
                        # parameters that are in the current state_dict.
                        named_parameters = dict(module.named_parameters(prefix=prefix[:-1], recurse=False))
                        params_to_gather = [named_parameters[k] for k in state_dict.keys() if k in named_parameters]
                        if len(params_to_gather) > 0:
                            # because zero3 puts placeholders in model params, this context
                            # manager gathers (unpartitions) the params of the current layer, then loads from
                            # the state dict and then re-partitions them again
                            with deepspeed.zero.GatheredParameters(params_to_gather, modifier_rank=0):
                                if torch.distributed.get_rank() == 0:
                                    module._load_from_state_dict(*args)
                    else:
                        module._load_from_state_dict(*args)

                for name, child in module._modules.items():
                    if child is not None:
                        load(child, state_dict, prefix + name + ".")

            load(model, state_dict)
            incompatible_keys = []
        return incompatible_keys
    def feature_select(self, image_forward_outs):

        if self.select_layer>100:
            image_features = image_forward_outs[-4:]
        else:
            image_features = image_forward_outs[-1]
        return image_features

    def forward_features(self, x):
        x = self.vision_tower.stem(x)
        image_forward_out=[]
        for blk in self.vision_tower.stages:
            x = blk(x)
            b,c,h,w=x.shape
            image_forward_out.append(x.view(b,c,-1).transpose(1,2))
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
                image_feature = self.feature_select(image_forward_out)
                image_features.append(image_feature)
        else:
            image_forward_outs = self.forward_features(images.to(device=self.device, dtype=self.dtype))
            image_features = self.feature_select(image_forward_outs)

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
        return 40 if 'xxlarge' in self.vision_tower_name else 36 ## use for layer decay
    @property
    def hidden_size(self):
        return self.vision_tower.hidden_size

    @property
    def num_patches(self):
        return (cfg['image_size'] // self.patch_embed.patch_size[0]) ** 2
