import torch.nn as nn
import torch.nn.functional as F
from transformers.adapters import AdapterConfig
from transformers.adapters.modeling import Adapter

class SeemAdapter(nn.Module):
    def __init__(self, d_model, kernel_size=3, channel_depth=64, reduction=4, num_adapters=1):
        super().__init__()
        # self.conv = nn.Conv1d(d_model, channel_depth, kernel_size=kernel_size, padding=1)
        
        # self.linear = nn.Linear(channel_depth, d_model)
        # self.layer_norm_conv = nn.LayerNorm(d_model)

        down_sample = d_model // reduction
        
        self.adapters = nn.ModuleList([
            Adapter(
                adapter_name="pfeiffer",
                config=AdapterConfig.load("pfeiffer"),
                input_size=d_model,
                down_sample=down_sample
            ) for _ in range(num_adapters)
        ])
        
        # self.adapter_norms = nn.ModuleList([
        #     nn.LayerNorm(d_model) for _ in range(num_adapters)
        # ])


    def forward(self, x, residual_input):
        # x = x.transpose(0, 1)       # [batch_size, sequence_length, embed_dim]
        # x = x.transpose(1, 2)       # [batch_size, embed_dim, sequence_length]
        # x = F.relu(self.conv(x))    # [batch_size, channel_depth, sequence_length]
        # x = x.transpose(1, 2)       # [batch_size, sequence_length, channel_depth]
        # x = self.linear(x)          # [batch_size, sequence_length, embed_dim]
        # x = self.layer_norm_conv(x)
        # x = x.transpose(0, 1) 

        # for adapter, layer_norm in zip(self.adapters, self.adapter_norms):
        #     x, *_ = adapter(x, residual_input=residual_input)
        #     x = layer_norm(x)

        for adapter in self.adapters:
            x, *_ = adapter(x, residual_input=residual_input)
        
        return x
