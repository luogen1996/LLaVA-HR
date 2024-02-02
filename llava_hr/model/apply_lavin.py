from dataclasses import dataclass, field
from typing import Dict, Optional, Sequence, List
import torch
from torch import nn
from typing import Optional, Tuple
from  torch.cuda.amp import autocast
import torch.nn.functional as F
import transformers
from transformers.models.llama.modeling_llama import LlamaDecoderLayer,LlamaRMSNorm
from transformers.models.clip.modeling_clip import CLIPEncoderLayer,CLIPVisionConfig
from transformers import LlamaConfig,CLIPVisionConfig



class Adapter_Router_plus(nn.Module):
    def __init__(
            self,
            config,
            lavin_in_features=512,
            enable_clip=False
    ):
        super().__init__()
        if enable_clip:
            hidden_dim=8
        else:
            hidden_dim=config.lavin_hidden_dim
        self.conv_A=nn.ModuleList([nn.Conv1d(lavin_in_features,hidden_dim,1,groups=1,bias=True) for i in range(config.lavin_n_router)])


        self.conv_B = nn.ModuleList([nn.Conv1d(hidden_dim, lavin_in_features, 1, groups=config.lavin_groups, bias=True) for i in range(config.lavin_n_router)])


        self.expert_weights=nn.Linear(lavin_in_features,config.lavin_n_router)

        self.dropout=nn.Dropout(0.1)
        self.groups=config.lavin_groups
        self.scale=config.lavin_scale
        self.t=config.lavin_t


        for conv in self.conv_A:
            nn.init.xavier_uniform_(conv.weight)
            nn.init.zeros_(conv.bias)
        for conv in self.conv_B:
            nn.init.zeros_(conv.weight)
            nn.init.zeros_(conv.bias)

    def forward(self, x,weights=None):
        src_type = x.dtype
        dst_type = self.expert_weights.weight.dtype
        if weights is None:
            weights=torch.softmax(self.expert_weights(x[:,0].type(dst_type))/self.t,-1)
        x=x.transpose(1,2)
        out=0
        for i,conv in enumerate(self.conv_A):
            out+=self.conv_B[i](F.silu(self.dropout(self.conv_A[i](x.type(dst_type)))))*self.scale*weights[:,i,None,None]
        out=out.type(src_type)+x
        x=out.transpose(1,2).contiguous()
        return x

def _copy_weight(src_module,dst_module):
    '''
    :param src_module: new weights
    :param dst_module: pre-trained model weights
    :return:
    '''
    for name, module in dst_module.named_children():
        assert hasattr(src_module,name)
        self_module = getattr(src_module, name)
        if isinstance(module, nn.Linear) or isinstance(module, LlamaRMSNorm) or isinstance(module, nn.LayerNorm):
            self_module.weight = module.weight
            if hasattr(module, "bias"):
                self_module.bias = module.bias
        else:
            _copy_weight(self_module,module)

def replace_module(model, enc_module,dec_mocule):
    device = next(model.parameters()).device
    if enc_module is not None:
        enc_module = enc_module.to(device)  # 将enc_module移至模型的设备
    if dec_mocule is not None:
        dec_mocule = dec_mocule.to(device)  # 将dec_mocule移至模型的设备
    for name, _ in model.named_children():
        if hasattr(_,'layers') and isinstance(_.layers,nn.ModuleList):
            if 'vision_tower' in name:
                if enc_module is not None:
                    _.layers=enc_module
            else:
                if dec_mocule is not None:
                    _.layers = dec_mocule
        else:
            replace_module(_, enc_module,dec_mocule)

class LaVINCLIPLayer(CLIPEncoderLayer):
    def __init__(self, config: CLIPVisionConfig, lavin_config):
        super().__init__(config)

        self.adapter=Adapter_Router_plus(config=lavin_config,enable_clip=True,lavin_in_features=config.hidden_size)

    @classmethod
    def from_pretrained(cls,enc_module, config, lavin_config):
        assert isinstance(enc_module, CLIPEncoderLayer)
        lavin_clip_layer = cls(config, lavin_config)
        _copy_weight(lavin_clip_layer,enc_module)
        return lavin_clip_layer

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: torch.Tensor,
        causal_attention_mask: torch.Tensor,
        output_attentions: Optional[bool] = False,
    ) -> Tuple[torch.FloatTensor]:
        """
        Args:
            hidden_states (`torch.FloatTensor`): input to the layer of shape `(batch, seq_len, embed_dim)`
            attention_mask (`torch.FloatTensor`): attention mask of size
                `(batch, 1, tgt_len, src_len)` where padding elements are indicated by very large negative values.
                `(config.encoder_attention_heads,)`.
            output_attentions (`bool`, *optional*):
                Whether or not to return the attentions tensors of all attention layers. See `attentions` under
                returned tensors for more detail.
        """
        residual = hidden_states

        hidden_states = self.layer_norm1(hidden_states)

        # lavin clip adapter
        hidden_states=self.adapter(hidden_states)


        hidden_states, attn_weights = self.self_attn(
            hidden_states=hidden_states,
            attention_mask=attention_mask,
            causal_attention_mask=causal_attention_mask,
            output_attentions=output_attentions,
        )
        hidden_states = residual + hidden_states

        residual = hidden_states
        hidden_states = self.layer_norm2(hidden_states)
        hidden_states = self.mlp(hidden_states)
        hidden_states = residual + hidden_states

        outputs = (hidden_states,)

        if output_attentions:
            outputs += (attn_weights,)

        return outputs


class LaVINDecoderLayer(LlamaDecoderLayer):
    def __init__(self, config: LlamaConfig, lavin_config):
        super().__init__(config)

        self.adapter = Adapter_Router_plus(config=lavin_config,lavin_in_features=config.hidden_size)

        self.cache_weights = torch.zeros(
            (100, 2)
        ).cuda()

    @classmethod
    def from_pretrained(cls,dec_module, config, lavin_config):
        assert isinstance(dec_module, LlamaDecoderLayer)
        lavin_dec_layer = cls(config, lavin_config)
        _copy_weight(lavin_dec_layer,dec_module)
        return lavin_dec_layer

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_value: Optional[Tuple[torch.Tensor]] = None,
        output_attentions: Optional[bool] = False,
        use_cache: Optional[bool] = False,
    ) -> Tuple[torch.FloatTensor, Optional[Tuple[torch.FloatTensor, torch.FloatTensor]]]:
        """
        Args:
            hidden_states (`torch.FloatTensor`): input to the layer of shape `(batch, seq_len, embed_dim)`
            attention_mask (`torch.FloatTensor`, *optional*): attention mask of size
                `(batch, 1, tgt_len, src_len)` where padding elements are indicated by very large negative values.
            output_attentions (`bool`, *optional*):
                Whether or not to return the attentions tensors of all attention layers. See `attentions` under
                returned tensors for more detail.
            use_cache (`bool`, *optional*):
                If set to `True`, `past_key_values` key value states are returned and can be used to speed up decoding
                (see `past_key_values`).
            past_key_value (`Tuple(torch.FloatTensor)`, *optional*): cached past key and value projection states
        """

        residual = hidden_states

        hidden_states = self.input_layernorm(hidden_states)

        # lavin adapter
        if self.training:
            weights = torch.softmax(self.adapter.expert_weights(hidden_states[:, 1]) / self.adapter.t, -1).type_as(residual)
        else:
            src_type = hidden_states.dtype
            dst_type = self.adapter.expert_weights.weight.dtype
            if past_key_value is None:
                self.cache_weights=self.cache_weights.type(dst_type).to(residual.device)
                self.cache_weights[:residual.shape[0]] = torch.softmax(self.adapter.expert_weights(hidden_states[:, 1].type(dst_type)) / self.adapter.t, -1)
            weights=self.cache_weights[:residual.shape[0]].type(src_type)

        hidden_states=self.adapter(hidden_states,weights)

        # Self Attention
        hidden_states, self_attn_weights, present_key_value = self.self_attn(
            hidden_states=hidden_states,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_value=past_key_value,
            output_attentions=output_attentions,
            use_cache=use_cache,
        )
        hidden_states = residual + hidden_states

        # Fully Connected
        residual = hidden_states
        hidden_states = self.post_attention_layernorm(hidden_states)
        hidden_states = self.mlp(hidden_states)
        hidden_states = residual + hidden_states

        outputs = (hidden_states,)

        if output_attentions:
            outputs += (self_attn_weights,)

        if use_cache:
            outputs += (present_key_value,)

        return outputs


def set_lavin(model, config, training_config, learnable_keys=['adapter'], ignored_key='projector'):
    """
    A function to insert LaVIN adapters and set parameters to trainable.

    :param model: The model whose layers to replace.
    :param config: The configuration object with the details for pretrained models.
    :param training_config: The configuration object specifying training parameters.
    :param learnable_keys: List of keys specifying which layers of the model to retain for training.
    :param ignored_key: The key for the layer to be ignored during training.
    :return: The model after replacement of layers and setting of certain layers' parameters to trainable.
    """

    # Load the pretrained config for the vision tower of the clip model and the llama model respectively
    clip_config = CLIPVisionConfig.from_pretrained(config.vision_tower)
    llama_config = transformers.AutoConfig.from_pretrained(config.model_name_or_path, trust_remote_code=True)

    # Initialize lists to store new decoder and encoder layers
    decoder_replacements = []
    encoder_replacements = []

    # Loop over the named modules in the model and replace them
    for name, module in model.named_modules():
        if isinstance(module, LlamaDecoderLayer):
            # If the module is a type of LlamaDecoderLayer, replace it
            new_dec = LaVINDecoderLayer.from_pretrained(module, llama_config, training_config)
            decoder_replacements.append(new_dec)

        if isinstance(module, CLIPEncoderLayer):
            # If the module is a type of CLIPEncoderLayer, replace it
            new_enc = LaVINCLIPLayer.from_pretrained(module, clip_config, training_config)
            encoder_replacements.append(new_enc)

    # Use the replace_module() function to replace old layers in model with new ones
    replace_module(model, nn.ModuleList(encoder_replacements), nn.ModuleList(decoder_replacements))

    total = 0
    trainable_names = []

    # Loop over parameters, set some to trainable and others to non-trainable based on keys
    for name, param in model.named_parameters():
        if ignored_key in name:
            total += param.nelement()
            trainable_names.append(name)
            continue
        else:
            for key in learnable_keys:
                if key in name:
                    total += param.nelement()
                    param.requires_grad = True
                    trainable_names.append(name)
                else:
                    # Set parameters to be non-trainable
                    param.requires_grad = False

    # Print the total number of trainable parameters
    # print('  + Number of trainable params: %.2fM' % (total / 1e6))

    return model

if __name__ == '__main__':
    global local_rank
    from llava_hr.train.train import ModelArguments, DataArguments, TrainingArguments
    parser = transformers.HfArgumentParser(
        (ModelArguments, DataArguments, TrainingArguments))
    model_args, data_args, training_args = parser.parse_args_into_dataclasses()
    local_rank = training_args.local_rank
    compute_dtype = (torch.float16 if training_args.fp16 else (torch.bfloat16 if training_args.bf16 else torch.float32))

    bnb_model_from_pretrained_args = {}
    model = transformers.LlamaForCausalLM.from_pretrained(
        model_args.model_name_or_path,
        cache_dir=training_args.cache_dir,
        **bnb_model_from_pretrained_args
    )
    w1=model.model.layers[0].self_attn.q_proj.weight
    set_lavin(model,model_args,training_args)
    w2=model.model.layers[0].self_attn.q_proj.weight
    print((w1-w2).sum())