
from spikingjelly.activation_based import surrogate
import math
import torch
import torch.nn as nn
from robomimic.models.base_nets import ConvBase
from spikingjelly.activation_based import functional
from spikingjelly.activation_based.neuron import LIFNode
def conv3x3(in_planes, out_planes, stride=1, groups=1, dilation=1):
    """3x3 convolution with padding"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=dilation, groups=groups, bias=False, dilation=dilation)


def conv1x1(in_planes, out_planes, stride=1):
    """1x1 convolution"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride, bias=False)


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None, groups=1,
                 base_width=64, dilation=1, norm_layer=None, single_step_neuron: callable = None, **kwargs):
        super(BasicBlock, self).__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        if groups != 1 or base_width != 64:
            raise ValueError('BasicBlock only supports groups=1 and base_width=64')
        if dilation > 1:
            raise NotImplementedError("Dilation > 1 not supported in BasicBlock")
        # Both self.conv1 and self.downsample layers downsample the input when stride != 1
        self.conv1 = conv3x3(inplanes, planes, stride)
        self.bn1 = norm_layer(planes)
        self.sn1 = single_step_neuron(**kwargs)
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = norm_layer(planes)
        self.sn2 = single_step_neuron(**kwargs)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.sn1(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.sn2(out)

        return out


class MultiStepBasicBlock(BasicBlock):
    def __init__(self, inplanes, planes, stride=1, downsample=None, groups=1,
                 base_width=64, dilation=1, norm_layer=None, multi_step_neuron: callable = None, **kwargs):
        super().__init__(inplanes, planes, stride, downsample, groups,
                 base_width, dilation, norm_layer, multi_step_neuron, **kwargs)

    def forward(self, x_seq):
        identity = x_seq

        out = functional.seq_to_ann_forward(x_seq, [self.conv1, self.bn1])
        out = self.sn1(out)

        out = functional.seq_to_ann_forward(out, [self.conv2, self.bn2])

        if self.downsample is not None:
            identity = functional.seq_to_ann_forward(x_seq, self.downsample)

        out += identity
        out = self.sn2(out)

        return out


class MultiStepSpikingResNet(nn.Module):    
    def __init__(self, block, layers, num_classes=1000, zero_init_residual=False,
                 groups=1, width_per_group=64, replace_stride_with_dilation=None,
                 norm_layer=None, T: int = None, spiking_neuron: callable = None, **kwargs):
        """
        初始化多步脉冲残差网络
        
        Args:
            block: 残差块类型 (如BasicBlock, Bottleneck)
            layers: 每个阶段的块数量列表 [layer1_blocks, layer2_blocks, layer3_blocks, layer4_blocks]
            num_classes: 分类数量，默认为1000 (ImageNet标准)
            zero_init_residual: 是否将残差分支的最后一个BN层初始化为0，提高训练稳定性
            groups: 分组卷积的组数，默认为1
            width_per_group: 每组的基础宽度，默认为64
            replace_stride_with_dilation: 是否用空洞卷积替换步长，用于提高感受野
            norm_layer: 归一化层类型，默认为BatchNorm2d
            T: 时间步数，用于多步脉冲网络
            spiking_neuron: 多步脉冲神经元类型
            **kwargs: 传递给神经元的额外参数
        """
        super().__init__()
        
        # 存储时间步数
        self.T = T
        
        # 设置归一化层，默认为BatchNorm2d
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        self._norm_layer = norm_layer

        # 初始化基础参数
        self.inplanes = 64  # 初始输入通道数
        self.dilation = 1   # 初始空洞率
        
        # 设置空洞卷积替换策略
        if replace_stride_with_dilation is None:
            # 每个元素表示是否用空洞卷积替换2x2步长
            replace_stride_with_dilation = [False, False, False]
        if len(replace_stride_with_dilation) != 3:
            raise ValueError("replace_stride_with_dilation should be None "
                             "or a 3-element tuple, got {}".format(replace_stride_with_dilation))
        
        # 设置分组卷积参数
        self.groups = groups
        self.base_width = width_per_group
        # 第一层卷积：7x7卷积，步长为2，将输入从3通道转换为64通道
        self.conv1 = nn.Conv2d(3, self.inplanes, kernel_size=7, stride=2, padding=3,
                               bias=False)
        self.bn1 = norm_layer(self.inplanes)  # 批归一化
        self.sn1 = spiking_neuron(**kwargs)  # 脉冲神经元
        
        # 最大池化层：3x3池化，步长为2，进一步降低空间分辨率
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        
        # 构建四个残差层，每个层包含不同数量的残差块
        # layer1: 64通道，不改变空间分辨率
        self.layer1 = self._make_layer(block, 64, layers[0], spiking_neuron=spiking_neuron, **kwargs)
        # layer2: 128通道，空间分辨率减半
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2,
                                       dilate=replace_stride_with_dilation[0], spiking_neuron=spiking_neuron,
                                       **kwargs)
        # layer3: 256通道，空间分辨率减半
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2,
                                       dilate=replace_stride_with_dilation[1], spiking_neuron=spiking_neuron,
                                       **kwargs)
        # layer4: 512通道，空间分辨率减半
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2,
                                       dilate=replace_stride_with_dilation[2], spiking_neuron=spiking_neuron,
                                       **kwargs)
        
        # 全局平均池化和全连接层（用于分类任务）
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(512 * block.expansion, num_classes)

        # 权重初始化
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                # 卷积层使用Kaiming初始化
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                # 归一化层权重初始化为1，偏置初始化为0
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

        # 残差分支的零初始化策略
        # 将每个残差块中最后一个BN层的权重初始化为0
        # 这样残差分支开始时输出为0，每个残差块表现为恒等映射
        # 根据论文 https://arxiv.org/abs/1706.02677，这可以提高模型性能0.2~0.3%
        if zero_init_residual:
            for m in self.modules():
                if isinstance(m, Bottleneck):
                    nn.init.constant_(m.bn3.weight, 0)
                elif isinstance(m, BasicBlock):
                    nn.init.constant_(m.bn2.weight, 0)

    def _make_layer(self, block, planes, blocks, stride=1, dilate=False, spiking_neuron: callable = None, **kwargs):
        """
        构建一个残差层，包含多个残差块
        
        Args:
            block: 残差块类型
            planes: 输出通道数
            blocks: 该层包含的残差块数量
            stride: 步长，用于下采样
            dilate: 是否使用空洞卷积
            spiking_neuron: 脉冲神经元类型
            **kwargs: 传递给神经元的额外参数
            
        Returns:
            nn.Sequential: 包含多个残差块的序列模块
        """
        norm_layer = self._norm_layer
        downsample = None
        previous_dilation = self.dilation
        
        # 处理空洞卷积
        if dilate:
            self.dilation *= stride
            stride = 1
            
        # 当步长不为1或输入输出通道数不匹配时，需要下采样分支
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                conv1x1(self.inplanes, planes * block.expansion, stride),  # 1x1卷积调整通道数和分辨率
                norm_layer(planes * block.expansion),  # 归一化
            )

        # 构建残差块列表
        layers = []
        # 第一个块可能需要下采样
        layers.append(block(self.inplanes, planes, stride, downsample, self.groups,
                            self.base_width, previous_dilation, norm_layer, spiking_neuron, **kwargs))
        self.inplanes = planes * block.expansion
        
        # 后续块不需要下采样
        for _ in range(1, blocks):
            layers.append(block(self.inplanes, planes, groups=self.groups,
                                base_width=self.base_width, dilation=self.dilation,
                                norm_layer=norm_layer, multi_step_neuron=spiking_neuron, **kwargs))

        return nn.Sequential(*layers)

    def _forward_impl(self, x: torch.Tensor):
        """
        前向传播的核心实现
        
        Args:
            x: 输入张量，形状为 [N, C, H, W] 或 [T, N, C, H, W]
            
        Returns:
            torch.Tensor: 输出特征，形状为 [N, C, H, W]
        """
        x_seq = None
        
        # 处理输入格式
        if x.dim() == 5:
            # 输入已经是多时间步格式 [T, N, C, H, W]
            x_seq = functional.seq_to_ann_forward(x, [self.conv1, self.bn1])
        else:
            # 输入是单时间步格式 [N, C, H, W]，需要扩展为多时间步
            assert self.T is not None, 'When x.shape is [N, C, H, W], self.T can not be None.'
            x = self.conv1(x)  # 第一层卷积
            x = self.bn1(x)    # 批归一化
            x.unsqueeze_(0)    # 添加时间维度
            x_seq = x.repeat(self.T, 1, 1, 1, 1)  # 复制到T个时间步
        
        # 通过脉冲神经元
        x_seq = self.sn1(x_seq)
        
        # 最大池化
        x_seq = functional.seq_to_ann_forward(x_seq, self.maxpool)

        # 通过四个残差层
        x_seq = self.layer1(x_seq)  # 64通道
        x_seq = self.layer2(x_seq)  # 128通道
        x_seq = self.layer3(x_seq)  # 256通道
        x_seq = self.layer4(x_seq)  # 512通道

        # 注释掉的部分是用于分类任务的全局平均池化和全连接层
        # x_seq = functional.seq_to_ann_forward(x_seq, self.avgpool)
        # x_seq = torch.flatten(x_seq, 2)
        # x_seq = functional.seq_to_ann_forward(x_seq, self.fc)
        
        # 对时间维度取平均，返回最终特征
        return x_seq.mean(0)

    def forward(self, x):
        """
        前向传播接口
        
        Args:
            x: 输入张量，形状为 [N, C, H, W] 或 [T, N, C, H, W]
            
        Returns:
            torch.Tensor: 输出特征
        """
        return self._forward_impl(x)

class SpikingResNet18(ConvBase):
    """
    A ResNet18 block that can be used to process input images.
    """
    def __init__(
        self,
        T: int = 4,
        input_channel: int = 3,
        multi_step_neuron: callable = LIFNode,
        surrogate_function: callable = surrogate.ATan(),
        step_mode: str = "m",
        v_threshold: float = 0.5,
        v_reset: float = 0.0,
        pretrained: bool = False,
        input_coord_conv: bool = False,
    ):
        """
        Args:
            input_channel (int): number of input channels for input images to the network.
                If not equal to 3, modifies first conv layer in ResNet to handle the number
                of input channels.
            pretrained (bool): if True, load pretrained weights for all ResNet layers.
            input_coord_conv (bool): if True, use a coordinate convolution for the first layer
                (a convolution where input channels are modified to encode spatial pixel location)
        """
        super().__init__()                               
        net = MultiStepSpikingResNet(block=MultiStepBasicBlock, 
                                        layers=[2, 2, 2, 2], 
                                        T=T, 
                                        spiking_neuron=multi_step_neuron,
                                        surrogate_function=surrogate_function,
                                        step_mode=step_mode,
                                        v_threshold=v_threshold,
                                        v_reset=v_reset)

        if pretrained:
            # TODO: load pretrained weights
            raise NotImplementedError("Pretrained weights are not available for Spiking ResNet")
        

        # cut the last fc layer
        self._input_coord_conv = input_coord_conv
        self._input_channel = input_channel
        # self.nets = torch.nn.Sequential(*(list(net.children())[:-2]))
        self.nets = net

    def output_shape(self, input_shape=None):
        """
        Function to compute output shape from inputs to this module. 

        Args:
            input_shape (iterable of int): shape of input. Does not include batch dimension.
                Some modules may not need this argument, if their output does not depend 
                on the size of the input, or if they assume fixed size input.

        Returns:
            out_shape ([int]): list of integers corresponding to output shape
        """
        assert(len(input_shape) == 3)
        out_h = int(math.ceil(input_shape[1] / 32.))
        out_w = int(math.ceil(input_shape[2] / 32.))
        return [512, out_h, out_w]

    def __repr__(self):
        """Pretty print network."""
        header = '{}'.format(str(self.__class__.__name__))
        return header + '(input_channel={}, input_coord_conv={})'.format(self._input_channel, self._input_coord_conv)
