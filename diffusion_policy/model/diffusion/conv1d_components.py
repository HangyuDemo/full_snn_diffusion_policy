import torch
import torch.nn as nn
from spikingjelly.activation_based import neuron, surrogate, layer
from diffusion_policy.model.diffusion.multiple_threshold import MultiThresholdLIFSpike, MultiThresholdLIFNode
from diffusion_policy.model.diffusion.single_threshold import LIFSpike
threshold = 1


class Downsample1d(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.snn_conv = layer.Conv1d(dim, dim, 3, padding=3 // 2, step_mode="m")

    def forward(self, x):
        return self.snn_conv(x)

class Upsample1d(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.snn_conv = layer.ConvTranspose1d(dim, dim, 3, padding=3 // 2, step_mode="m")

    def forward(self, x):
        return self.snn_conv(x)

class Conv1dBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, LCMT=True, n_groups=8):
        super().__init__()
        self.snn_conv = layer.Conv1d(in_channels, out_channels, kernel_size, padding=kernel_size // 2, step_mode="m")
        self.group_norm = layer.GroupNorm(n_groups, out_channels, step_mode="m")
        if LCMT:
            # self.lif_node = MultiThresholdLIFSpike(in_channels, num_thresholds=4, v_min=0.0, v_max=2.0)
            # self.lif_node = LIFSpike(threshold, False)
            # self.lif_node = MultiThresholdLIFNode(surrogate_function=surrogate.ATan(), step_mode="m",v_threshold=0.5)
            self.lif_node = neuron.LIFNode(surrogate_function=surrogate.ATan(), step_mode="m",v_threshold=0.5)
        else:
            self.lif_node = neuron.LIFNode(surrogate_function=surrogate.ATan(), step_mode="m",v_threshold=0.5)
    def forward(self, x):
        x = self.lif_node(x)
        x = self.snn_conv(x)
        x = self.group_norm(x)
        return x