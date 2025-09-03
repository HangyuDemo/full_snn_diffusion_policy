import torch
import torch.nn as nn

aa = 0.5
tau = 0.25
kappa = 0.5

# ==== Surrogate gradient ====
class SpikeAct(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input):
        ctx.save_for_backward(input)
        return (input > 0).float()

    @staticmethod
    def backward(ctx, grad_output):
        input, = ctx.saved_tensors
        grad_input = grad_output.clone()
        hu = (input.abs() < aa).float() / (2 * aa)
        return grad_input * hu

spikeAct = SpikeAct.apply

# ==== LIF state update ====
def state_update(u_t_n1, o_t_n1, W_mul_o_t1_n1, thre):
    u_t1_n1 = tau * u_t_n1 * (1 - o_t_n1) + W_mul_o_t1_n1 #([64, 64, 42, 42])
    v_t1_n1 = u_t1_n1 - thre
    o_t1_n1 = spikeAct(v_t1_n1)
    return u_t1_n1, o_t1_n1

# ==== LIFSpike Module ====
class LIFSpike(nn.Module):
    def __init__(self, kappa=0.5, dim=True):
        super(LIFSpike, self).__init__()
        self.dim = dim
        self.kappa = kappa
        self.w = None

        self.mem = None
        self.initialized = False

    def reset(self):
        self.mem = None
        self.initialized = False

    def forward(self, x):
        """
        x: shape [T, B, C, H, W] or [T, B, C]
        """
        # 自动获取设备，默认使用 GPU 如果可用，否则使用 CPU
        device = x.device if x.is_cuda else torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        
        if self.mem is not None:
            self.mem = self.mem.to(device)

        # 自动获取 in_channels (即第三个维度 C)
        in_channels = x.shape[2]  # 第三个维度对应通道数 C
        
        # 初始化权重
        if self.w is None:
            init_w = torch.full((in_channels,), self.kappa, dtype=torch.float32, device=device)
            self.w = nn.Parameter(init_w, requires_grad=True)  # [64]
            self.w.to(device)

        steps = x.shape[0]  # 2
        out = torch.zeros_like(x, device=device)  # Ensure output is on the same device as x
        k = self.w.unsqueeze(0).unsqueeze(-1).to(device)  # [1, 64, 1]
        if self.dim:
            k = k.unsqueeze(-1)
        k = k.tanh()  # [1, 64, 1]

        # 使用持久化 self.mem
        if not self.initialized or self.mem is None or self.mem.shape != x.shape[1:]:
            self.mem = torch.zeros(x.shape[1:], device=device, dtype=x.dtype)
            self.initialized = True

        for step in range(steps):
            prev_spike = out[step - 1] if step > 0 else torch.zeros_like(out[0], device=device)
            self.mem, out[step] = state_update(self.mem, prev_spike, x[step].to(device), k)

        return out
