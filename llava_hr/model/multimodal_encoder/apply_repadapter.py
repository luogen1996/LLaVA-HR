import timm
from typing import Callable, Optional, Tuple, Union
import  torch.nn as nn
import torch
class RepAdapter(nn.Module):

    """ Pytorch Implemention of RepAdapter for 1d tensor"""

    def __init__(
            self,
            in_features=1024,
            hidden_dim=8,
            groups=2,
            scale=1
    ):
        super().__init__()
        self.conv_A=nn.Conv1d(in_features,hidden_dim,1,groups=1,bias=True)
        self.conv_B = nn.Conv1d(hidden_dim, in_features, 1, groups=groups, bias=True)
        self.dropout=nn.Dropout(0.1)
        self.groups=groups
        self.scale=scale

        nn.init.xavier_uniform_( self.conv_A.weight)
        nn.init.zeros_(self.conv_A.bias)
        nn.init.zeros_(self.conv_B.weight)
        nn.init.zeros_(self.conv_B.bias)
    def forward(self, x):
        x=x.transpose(1,2)
        x=self.conv_B(self.dropout(self.conv_A(x)))*self.scale+x
        x=x.transpose(1,2).contiguous()
        return x

def forward_vit_block_adapter(self, x, rope: Optional[torch.Tensor] = None, attn_mask: Optional[torch.Tensor] = None):
    if self.gamma_1 is None:
        x = x + self.drop_path1(self.attn(self.align_stages_attn(self.norm1(x)), rope=rope, attn_mask=attn_mask))
        x = x + self.drop_path2(self.mlp(self.norm2(x)))
    else:
        x = x + self.drop_path1(self.gamma_1 * self.attn(self.align_stages_attn(self.norm1(x)), rope=rope, attn_mask=attn_mask))
        x = x + self.drop_path2(self.gamma_2 * self.mlp(self.norm2(x)))
    return x

def set_RepAdapter(model,dim=8, s=1):

    for _ in model.children():
        if type(_) == timm.models.eva.EvaBlock:
            _.align_stages_attn = RepAdapter(hidden_dim=dim, scale=s)
            _.s = s
            bound_method = forward_vit_block_adapter.__get__(_, _.__class__)
            setattr(_, 'forward', bound_method)
        elif len(list(_.children())) != 0:
            set_RepAdapter(_, dim, s)
