import torch
import torch.nn as nn
from typing import Callable, Optional
from spikingjelly.activation_based.neuron import BaseNode
# 如果你在用 spikingjelly 的 BaseNode，请从库里引入：
# from spikingjelly.activation_based.neuron import BaseNode, surrogate
# 这里给个极简的 BaseNode 占位（你项目里请删掉这段占位，直接继承你自己的 BaseNode）

import torch
import torch.nn as nn
import torch.nn.functional as F

tau = 0.25
beta = 10.0  # 控制 sigmoid 平滑程度

# ==== Surrogate gradient ====
class SpikeAct(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input):
        ctx.save_for_backward(input)
        return (input > 0).float()

    @staticmethod
    def backward(ctx, grad_output):
        input, = ctx.saved_tensors
        grad = grad_output.clone()
        mask = (input.abs() < 0.5).float()  # 梯度窗口 [-0.5, 0.5]
        return grad * mask

spikeAct = SpikeAct.apply

# ==== Soft multithreshold update ====
def multithreshold_state_update(u_prev, o_prev, input_t, thresholds, beta=10.0):
    u_next = tau * u_prev * (1 - o_prev) + input_t
    u_expand = u_next.unsqueeze(-1)
    thresholds = thresholds.view(1, 1, 1, -1).to(u_next.device)

    soft_spikes = torch.sigmoid(beta * (u_expand - thresholds)).sum(dim=-1)
    return u_next, soft_spikes

# ==== 可学习多阈值 LIF 神经元 ====
class MultiThresholdLIFSpike(nn.Module):
    def __init__(self, C, num_thresholds=4, dim=True, v_min=0.0, v_max=1.0):
        super().__init__()
        self.dim = dim
        self.C = C
        self.v_min = v_min
        self.v_max = v_max
        self.num_thresholds = num_thresholds

        # 标记这个模块是有状态的
        self.requires_mem = True

        # 膜电位 mem
        self.mem = None
        self.initialized = False

        # 初始化可学习阈值
        def inverse_sigmoid(x):
            eps = 1e-6
            x = torch.clamp(x, eps, 1 - eps)
            return torch.log(x / (1 - x))

        values = torch.linspace(0.1, 0.9, num_thresholds)
        init_raw = inverse_sigmoid((values - v_min) / (v_max - v_min))
        self.v_thresholds_raw = nn.Parameter(init_raw, requires_grad=True)

    def reset(self):
        """
        重置膜电位，在 reset_net 或训练步之间调用
        """
        self.mem = None
        self.initialized = False

    def get_thresholds(self):
        return self.v_min + (self.v_max - self.v_min) * torch.sigmoid(self.v_thresholds_raw)

    def threshold_separation_loss(self, margin=0.05):
        thresholds = self.get_thresholds()
        diffs = thresholds[1:] - thresholds[:-1]
        loss = torch.clamp(margin - diffs, min=0).sum()
        return loss

    def forward(self, x):
        """
        x: [T, B, C, H, W] 或 [T, B, C]
        output: same shape, 表示每步的soft spike count
        """
        T = x.shape[0]
        out = torch.zeros_like(x)
        thresholds = self.get_thresholds().to(x.device)

        input_shape = x.shape[1:]  # B, C, ...
        if (not self.initialized) or (self.mem is None) or (self.mem.shape != input_shape):
            self.mem = torch.zeros(input_shape, device=x.device, dtype=x.dtype)
            self.initialized = True

        for t in range(T):
            prev_spike = out[t - 1] if t > 0 else torch.zeros_like(out[0])
            self.mem, out[t] = multithreshold_state_update(self.mem, prev_spike, x[t], thresholds, beta=beta)

        return out


# ========================= 多阈值 LIF Node =========================
class MultiThresholdLIFNode(BaseNode):
    """
    多阈值 LIF 神经元：
    - 通过 num_thresholds 个可学习阈值建模“多次放电/计数”
    - 训练：输出软计数（sum(sigmoid(beta*(v - theta_k)))），便于梯度传播
    - 评估：可选择输出硬计数（跨过多少个阈值）或任意阈值是否被跨过（0/1）
    - reset 规则兼容 hard/soft：
        * hard reset：跨过任意阈值则 v -> v_reset
        * soft reset：跨过的最高阈值会被减去（模仿单阈值 v -= v_threshold 的多阈值版本）
    """
    def __init__(self,
                 tau: float = 2.,
                 decay_input: bool = True,
                 # 多阈值参数
                 num_thresholds: int = 4,
                 v_min: float = 0.0,
                 v_max: float = 1.0,
                 beta: float = 10.0,                 # 软门的平滑强度
                 # 其余与 LIFNode 保持一致
                 v_threshold: float = 1.0,          # 仅占位，不参与放电判据
                 v_reset: Optional[float] = 0.0,    # None=soft reset, 非 None=hard reset
                 surrogate_function: Optional[Callable] = None,
                 detach_reset: bool = False,
                 step_mode: str = 's',
                 backend: str = 'torch',
                 store_v_seq: bool = False,
                 # 评估输出模式：'soft_count' | 'hard_count' | 'any'
                 eval_output: str = 'hard_count'):
        assert isinstance(tau, float) and tau > 1.
        super().__init__(v_threshold, v_reset, surrogate_function, detach_reset, step_mode, backend, store_v_seq)

        self.tau = tau
        self.decay_input = decay_input

        self.num_thresholds = num_thresholds
        self.v_min = float(v_min)
        self.v_max = float(v_max)
        self.beta = float(beta)
        assert eval_output in ('soft_count', 'hard_count', 'any')
        self.eval_output = eval_output

        # ===== 初始化可学习阈值（用 raw logits + sigmoid 压到 [v_min, v_max]）=====
        def inverse_sigmoid(x):
            eps = 1e-6
            x = torch.clamp(x, eps, 1 - eps)
            return torch.log(x / (1 - x))

        init_lin = torch.linspace(0.15, 0.85, num_thresholds)  # 均匀分布在 (v_min, v_max) 内
        init_raw = inverse_sigmoid((init_lin - self.v_min) / max(1e-6, (self.v_max - self.v_min)))
        self.v_thresholds_raw = nn.Parameter(init_raw, requires_grad=True)

    # --------- 工具函数 ----------
    def get_thresholds(self) -> torch.Tensor:
        """返回形如 [K] 的升序阈值张量，范围在 [v_min, v_max]"""
        th = self.v_min + (self.v_max - self.v_min) * torch.sigmoid(self.v_thresholds_raw)
        return torch.sort(th).values

    def threshold_separation_loss(self, margin: float = 0.05) -> torch.Tensor:
        """
        让相邻阈值至少相隔 margin；若间隔小于 margin，产生正的惩罚
        """
        th = self.get_thresholds()
        diffs = th[1:] - th[:-1]
        return torch.clamp(margin - diffs, min=0).sum()

    # --------- 膜电位积分（与 LIFNode 一致） ----------
    @staticmethod
    @torch.jit.script
    def _charge_decay_input_reset0(x: torch.Tensor, v: torch.Tensor, tau: float):
        # v = v + (x - v) / tau
        return v + (x - v) / tau

    @staticmethod
    @torch.jit.script
    def _charge_decay_input(x: torch.Tensor, v: torch.Tensor, v_reset: float, tau: float):
        # v = v + (x - (v - v_reset)) / tau
        return v + (x - (v - v_reset)) / tau

    @staticmethod
    @torch.jit.script
    def _charge_no_decay_input_reset0(x: torch.Tensor, v: torch.Tensor, tau: float):
        # v = v * (1 - 1/tau) + x
        return v * (1. - 1. / tau) + x

    @staticmethod
    @torch.jit.script
    def _charge_no_decay_input(x: torch.Tensor, v: torch.Tensor, v_reset: float, tau: float):
        # v = v - (v - v_reset)/tau + x
        return v - (v - v_reset) / tau + x

    def _charge(self, x: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
        """根据 decay_input / v_reset 与 LIFNode 同步的积分形式更新 v"""
        if self.decay_input:
            if self.v_reset is None or self.v_reset == 0.:
                return self._charge_decay_input_reset0(x, v, self.tau)
            else:
                return self._charge_decay_input(x, v, float(self.v_reset), self.tau)
        else:
            if self.v_reset is None or self.v_reset == 0.:
                return self._charge_no_decay_input_reset0(x, v, self.tau)
            else:
                return self._charge_no_decay_input(x, v, float(self.v_reset), self.tau)

    # --------- 放电/重置（多阈值） ----------
    def _spike_soft_count(self, v: torch.Tensor, thresholds: torch.Tensor) -> torch.Tensor:
        """
        软计数：sum_k sigmoid(beta * (v - th_k))
        输出与 v 同形状
        """
        # v[..., None] - [*shape, 1]; thresholds[None,...] - [1, K]
        return torch.sigmoid(self.beta * (v.unsqueeze(-1) - thresholds.view(*([1] * v.dim()), -1))).sum(dim=-1)

    def _spike_hard_mask_and_count(self, v: torch.Tensor, thresholds: torch.Tensor):
        """
        硬判据：mask_k = (v >= th_k)；count = sum_k mask_k；any = 0/1
        """
        mask = (v.unsqueeze(-1) >= thresholds.view(*([1]*v.dim()), -1)).to(v)
        count = mask.sum(dim=-1)
        any_spike = torch.clamp(count, max=1.0)
        return mask, count, any_spike

    def _soft_reset(self, v: torch.Tensor, thresholds: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        """
        软重置：减去“跨过的最高阈值”（若无跨阈值，则不减）
        mask: [*shape, K]
        """
        if mask.numel() == 0:
            return v
        # 把阈值广播成与 mask 同形状，然后取被跨越阈值中的最大值
        th_b = thresholds.view(*([1]*v.dim()), -1).expand_as(mask)
        crossed_th = mask * th_b
        # 最高被跨阈值（若未跨越则为 0）
        max_crossed = crossed_th.max(dim=-1).values
        return v - max_crossed

    # --------- Step 前端 ----------
    def single_step_forward(self, x: torch.Tensor):
        # 初始化/对齐 v 的形状
        self.v_float_to_tensor(x)

        # 先积分更新膜电位
        self.v = self._charge(x, self.v)

        thresholds = self.get_thresholds().to(self.v.device, self.v.dtype)

        if self.training:
            # 训练：输出软计数，reset 用硬/软逻辑与 v_reset 对齐
            spike_soft = self._spike_soft_count(self.v, thresholds)
            # 为了与单阈值 soft/hard reset 逻辑一致，这里按 v_reset 是否为 None 决定 reset
            if self.v_reset is None:
                # soft reset：用硬 mask 计算最高跨越阈值并减去（稳定）
                hard_mask, _, any_spike = self._spike_hard_mask_and_count(self.v, thresholds)
                self.v = self._soft_reset(self.v, thresholds, hard_mask)
            else:
                # hard reset：跨越任一阈值就置为 v_reset
                _, _, any_spike = self._spike_hard_mask_and_count(self.v, thresholds)
                self.v = float(self.v_reset) * any_spike + (1. - any_spike) * self.v
            return spike_soft
        else:
            # 评估：根据 eval_output 选择输出
            hard_mask, hard_count, any_spike = self._spike_hard_mask_and_count(self.v, thresholds)

            if self.eval_output == 'soft_count':
                out = self._spike_soft_count(self.v, thresholds)
            elif self.eval_output == 'hard_count':
                out = hard_count
            else:  # 'any'
                out = any_spike

            # 执行 reset
            if self.v_reset is None:
                self.v = self._soft_reset(self.v, thresholds, hard_mask)
            else:
                self.v = float(self.v_reset) * any_spike + (1. - any_spike) * self.v

            return out

    def multi_step_forward(self, x_seq: torch.Tensor):
        """
        x_seq: [T, ...]；返回与 x_seq 同形状
        """
        self.v_float_to_tensor(x_seq[0])
        T = x_seq.shape[0]
        out = torch.zeros_like(x_seq)
        if self.store_v_seq:
            self.v_seq = torch.zeros_like(x_seq)

        thresholds = self.get_thresholds().to(x_seq.device, x_seq.dtype)

        for t in range(T):
            # 积分
            self.v = self._charge(x_seq[t], self.v)

            if self.training:
                spike_soft = self._spike_soft_count(self.v, thresholds)
                # reset
                if self.v_reset is None:
                    hard_mask, _, _ = self._spike_hard_mask_and_count(self.v, thresholds)
                    self.v = self._soft_reset(self.v, thresholds, hard_mask)
                else:
                    _, _, any_spike = self._spike_hard_mask_and_count(self.v, thresholds)
                    self.v = float(self.v_reset) * any_spike + (1. - any_spike) * self.v
                out[t] = spike_soft
            else:
                hard_mask, hard_count, any_spike = self._spike_hard_mask_and_count(self.v, thresholds)
                if self.eval_output == 'soft_count':
                    out[t] = self._spike_soft_count(self.v, thresholds)
                elif self.eval_output == 'hard_count':
                    out[t] = hard_count
                else:
                    out[t] = any_spike

                if self.v_reset is None:
                    self.v = self._soft_reset(self.v, thresholds, hard_mask)
                else:
                    self.v = float(self.v_reset) * any_spike + (1. - any_spike) * self.v

            if self.store_v_seq:
                self.v_seq[t] = self.v

        return out
