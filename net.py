import torch
import torch.nn as nn
import math
import torch.nn.functional as F
import torch.utils.checkpoint as checkpoint
from timm.models.layers import DropPath, to_2tuple, trunc_normal_
from einops import rearrange


def drop_path(x, drop_prob: float = 0., training: bool = False):
    """
    Drop paths (Stochastic Depth) per sample (when applied in main path of residual blocks).
    This is the same as the DropConnect impl I created for EfficientNet, etc networks, however,
    the original name is misleading as 'Drop Connect' is a different form of dropout in a separate paper...
    See discussion: https://github.com/tensorflow/tpu/issues/494#issuecomment-532968956 ... I've opted for
    changing the layer and argument names to 'drop path' rather than mix DropConnect as a layer name and use
    'survival rate' as the argument.
    """
    if drop_prob == 0. or not training:
        return x
    keep_prob = 1 - drop_prob
    # work with diff dim tensors, not just 2D ConvNets
    shape = (x.shape[0],) + (1,) * (x.ndim - 1)
    random_tensor = keep_prob + \
                    torch.rand(shape, dtype=x.dtype, device=x.device)
    random_tensor.floor_()  # binarize
    output = x.div(keep_prob) * random_tensor
    return output


class DropPath(nn.Module):
    """
    Drop paths (Stochastic Depth) per sample  (when applied in main path of residual blocks).
    """

    def __init__(self, drop_prob=None):
        super(DropPath, self).__init__()
        self.drop_prob = drop_prob

    def forward(self, x):
        return drop_path(x, self.drop_prob, self.training)


class AttentionBase(nn.Module):
    def __init__(self,
                 dim,
                 num_heads=8,
                 qkv_bias=False, ):
        super(AttentionBase, self).__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = nn.Parameter(torch.ones(num_heads, 1, 1))
        self.qkv1 = nn.Conv2d(dim, dim * 3, kernel_size=1, bias=qkv_bias)
        self.qkv2 = nn.Conv2d(dim * 3, dim * 3, kernel_size=3, padding=1, bias=qkv_bias)
        self.proj = nn.Conv2d(dim, dim, kernel_size=1, bias=qkv_bias)

    def forward(self, x):
        # [batch_size, num_patches + 1, total_embed_dim]
        b, c, h, w = x.shape
        qkv = self.qkv2(self.qkv1(x))
        q, k, v = qkv.chunk(3, dim=1)
        q = rearrange(q, 'b (head c) h w -> b head c (h w)',
                      head=self.num_heads)
        k = rearrange(k, 'b (head c) h w -> b head c (h w)',
                      head=self.num_heads)
        v = rearrange(v, 'b (head c) h w -> b head c (h w)',
                      head=self.num_heads)
        q = torch.nn.functional.normalize(q, dim=-1)
        k = torch.nn.functional.normalize(k, dim=-1)
        # transpose: -> [batch_size, num_heads, embed_dim_per_head, num_patches + 1]
        # @: multiply -> [batch_size, num_heads, num_patches + 1, num_patches + 1]
        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)

        out = (attn @ v)

        out = rearrange(out, 'b head c (h w) -> b (head c) h w',
                        head=self.num_heads, h=h, w=w)

        out = self.proj(out)
        return out

class BaseFeatureExtraction(nn.Module):
    def __init__(self,
                 dim,
                 num_heads=8,  # 此参数在焦点调制中未使用，保留仅为接口兼容性
                 ffn_expansion_factor=1.,
                 qkv_bias=False,
                 mlp_ratio=4.,
                 drop=0.,
                 proj_drop=0.,
                 LayerNorm_type='WithBias'
                 ):
        super().__init__()
        self.norm1 = LayerNorm(dim, LayerNorm_type)
        self.modulation = FocalModulation(
            dim=dim,
            proj_drop=proj_drop
        )
        self.norm2 = LayerNorm(dim, LayerNorm_type)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(
            in_features=dim,
            hidden_features=mlp_hidden_dim,
            act_layer=nn.GELU,
            drop=drop
        )

    def forward(self, x):
        B, C, H, W = x.shape
        shortcut = x
        x = self.norm1(x)

        x = rearrange(x, 'b c h w -> b h w c')
        x = self.modulation(x)
        x = rearrange(x, 'b h w c -> b (h w) c')
        x = self.mlp(x)
        x = rearrange(x, 'b (h w) c -> b c h w', h=H, w=W)

        x = shortcut + x

        shortcut = x
        x = self.norm2(x)

        x = rearrange(x, 'b c h w -> b (h w) c')
        x = self.mlp(x)
        x = rearrange(x, 'b (h w) c -> b c h w', h=H, w=W)

        x = shortcut + x

        return x

#传统残差块：降维---卷积---升维
#倒置残差块：升维---卷积---降维
class InvertedResidualBlock(nn.Module):
    def __init__(self, inp, oup, expand_ratio):
        super(InvertedResidualBlock, self).__init__()
        hidden_dim = int(inp * expand_ratio)
        self.bottleneckBlock = nn.Sequential(
            # pw
            nn.Conv2d(inp, hidden_dim, 1, bias=False),
            # nn.BatchNorm2d(hidden_dim),
            nn.ReLU6(inplace=True),
            # dw
            nn.ReflectionPad2d(1),
            nn.Conv2d(hidden_dim, hidden_dim, 3, groups=hidden_dim, bias=False),
            # nn.BatchNorm2d(hidden_dim),
            nn.ReLU6(inplace=True),
            # pw-linear
            nn.Conv2d(hidden_dim, oup, 1, bias=False),
            # nn.BatchNorm2d(oup),
        )

    def forward(self, x):
        return self.bottleneckBlock(x)


class DetailNode(nn.Module):
    def __init__(self):
        super(DetailNode, self).__init__()
        # Scale is Ax + b, i.e. affine transformation
        self.theta_phi = InvertedResidualBlock(inp=32, oup=32, expand_ratio=2)
        self.theta_rho = InvertedResidualBlock(inp=32, oup=32, expand_ratio=2)
        self.theta_eta = InvertedResidualBlock(inp=32, oup=32, expand_ratio=2)
        self.shffleconv = nn.Conv2d(64, 64, kernel_size=1,
                                    stride=1, padding=0, bias=True)

    def separateFeature(self, x):
        z1, z2 = x[:, :x.shape[1] // 2], x[:, x.shape[1] // 2:x.shape[1]]
        return z1, z2

    def forward(self, z1, z2):
        z1, z2 = self.separateFeature(
            self.shffleconv(torch.cat((z1, z2), dim=1)))
        z2 = z2 + self.theta_phi(z1)
        z1 = z1 * torch.exp(self.theta_rho(z2)) + self.theta_eta(z2)
        return z1, z2


class DetailFeatureExtraction(nn.Module):
    def __init__(self, num_layers=3):
        super(DetailFeatureExtraction, self).__init__()
        INNmodules = [DetailNode() for _ in range(num_layers)]
        self.net = nn.Sequential(*INNmodules)

    def forward(self, x):
        z1, z2 = x[:, :x.shape[1] // 2], x[:, x.shape[1] // 2:x.shape[1]]
        for layer in self.net:
            z1, z2 = layer(z1, z2)
        return torch.cat((z1, z2), dim=1)


# =============================================================================

# =============================================================================
import numbers


##########################################################################
## Layer Norm
def to_3d(x):
    return rearrange(x, 'b c h w -> b (h w) c')


def to_4d(x, h, w):
    return rearrange(x, 'b (h w) c -> b c h w', h=h, w=w)


class BiasFree_LayerNorm(nn.Module):
    def __init__(self, normalized_shape):
        super(BiasFree_LayerNorm, self).__init__()
        if isinstance(normalized_shape, numbers.Integral):
            normalized_shape = (normalized_shape,)
        normalized_shape = torch.Size(normalized_shape)

        assert len(normalized_shape) == 1

        self.weight = nn.Parameter(torch.ones(normalized_shape))
        self.normalized_shape = normalized_shape

    def forward(self, x):
        sigma = x.var(-1, keepdim=True, unbiased=False)
        return x / torch.sqrt(sigma + 1e-5) * self.weight


class WithBias_LayerNorm(nn.Module):
    def __init__(self, normalized_shape):
        super(WithBias_LayerNorm, self).__init__()
        if isinstance(normalized_shape, numbers.Integral):
            normalized_shape = (normalized_shape,)
        normalized_shape = torch.Size(normalized_shape)

        assert len(normalized_shape) == 1

        self.weight = nn.Parameter(torch.ones(normalized_shape))

        self.bias = nn.Parameter(torch.zeros(normalized_shape))
        self.normalized_shape = normalized_shape

    def forward(self, x):

        mu = x.mean(-1, keepdim=True)

        sigma = x.var(-1, keepdim=True, unbiased=False)
        return (x - mu) / torch.sqrt(sigma + 1e-5) * self.weight + self.bias


class LayerNorm(nn.Module):
    def __init__(self, dim, LayerNorm_type):
        super(LayerNorm, self).__init__()
        if LayerNorm_type == 'BiasFree':
            self.body = BiasFree_LayerNorm(dim)
        else:
            self.body = WithBias_LayerNorm(dim)

    def forward(self, x):
        h, w = x.shape[-2:]
        return to_4d(self.body(to_3d(x)), h, w)


##########################################################################
## Gated-Dconv Feed-Forward Network (GDFN)
class FeedForward(nn.Module):
    def __init__(self, dim, ffn_expansion_factor, bias):
        super(FeedForward, self).__init__()

        hidden_features = int(dim * ffn_expansion_factor)

        self.project_in = nn.Conv2d(
            dim, hidden_features * 2, kernel_size=1, bias=bias)

        self.dwconv = nn.Conv2d(hidden_features * 2, hidden_features * 2, kernel_size=3,
                                stride=1, padding=1, groups=hidden_features * 2, bias=bias)

        self.project_out = nn.Conv2d(
            hidden_features, dim, kernel_size=1, bias=bias)

    def forward(self, x):
        x = self.project_in(x)
        x1, x2 = self.dwconv(x).chunk(2, dim=1)
        x = F.gelu(x1) * x2
        x = self.project_out(x)
        return x


##########################################################################
## Multi-DConv Head Transposed Self-Attention (MDTA)
class Attention(nn.Module):
    def __init__(self, dim, num_heads, bias):
        super(Attention, self).__init__()
        self.num_heads = num_heads
        self.temperature = nn.Parameter(torch.ones(num_heads, 1, 1))

        self.qkv = nn.Conv2d(dim, dim * 3, kernel_size=1, bias=bias)
        self.qkv_dwconv = nn.Conv2d(
            dim * 3, dim * 3, kernel_size=3, stride=1, padding=1, groups=dim * 3, bias=bias)
        self.project_out = nn.Conv2d(dim, dim, kernel_size=1, bias=bias)

    def forward(self, x):
        b, c, h, w = x.shape

        qkv = self.qkv_dwconv(self.qkv(x))
        q, k, v = qkv.chunk(3, dim=1)

        q = rearrange(q, 'b (head c) h w -> b head c (h w)',
                      head=self.num_heads)
        k = rearrange(k, 'b (head c) h w -> b head c (h w)',
                      head=self.num_heads)
        v = rearrange(v, 'b (head c) h w -> b head c (h w)',
                      head=self.num_heads)

        q = torch.nn.functional.normalize(q, dim=-1)
        k = torch.nn.functional.normalize(k, dim=-1)

        attn = (q @ k.transpose(-2, -1)) * self.temperature
        attn = attn.softmax(dim=-1)

        out = (attn @ v)

        out = rearrange(out, 'b head c (h w) -> b (head c) h w',
                        head=self.num_heads, h=h, w=w)

        out = self.project_out(out)
        return out

####替换Attention##########################################################################
'''
    替换逻辑：
        多尺度上下文聚合：通过不同层级的深度卷积（focal_layers）
        捕捉多尺度特征，替代自注意力的全局依赖计算。
        门控机制：利用 gates 参数动态融合不同层级的上下文信息，避免自注意力的二次复杂度。
        计算效率：焦点调制的时间复杂度为 O(N)，而自注意力为 O(N 2)，更适合高分辨率图像。
'''
class FocalModulation(nn.Module):
    """ Focal Modulation

    Args:
        dim (int): Number of input channels.
        proj_drop (float, optional): Dropout ratio of output. Default: 0.0
        focal_level (int): Number of focal levels
        focal_window (int): Focal window size at focal level 1
        focal_factor (int, default=2): Step to increase the focal window
        use_postln (bool, default=False): Whether use post-modulation layernorm
    """

    def __init__(self, dim, proj_drop=0., focal_level=2, focal_window=5, focal_factor=2, use_postln=True):

        super().__init__()
        self.dim = dim

        # specific args for focalv3
        self.focal_level = focal_level
        self.focal_window = focal_window
        self.focal_factor = focal_factor
        self.use_postln = use_postln

        self.f = nn.Linear(dim, 2 * dim + (self.focal_level + 1), bias=True)
        self.h = nn.Conv2d(dim, dim, kernel_size=1, stride=1, padding=0, groups=1, bias=True)

        self.act = nn.GELU()
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)
        self.focal_layers = nn.ModuleList()

        if self.use_postln:
            self.ln = nn.LayerNorm(dim)

        for k in range(self.focal_level):
            kernel_size = self.focal_factor * k + self.focal_window
            self.focal_layers.append(
                nn.Sequential(
                    nn.Conv2d(dim, dim, kernel_size=kernel_size, stride=1, groups=dim,
                              padding=kernel_size // 2, bias=False),
                    nn.GELU(),
                )
            )

    def forward(self, x):
        """ Forward function.

        Args:
            x: input features with shape of (B, H, W, C)
        """
        B, nH, nW, C = x.shape
        x = self.f(x)
        x = x.permute(0, 3, 1, 2).contiguous()
        q, ctx, gates = torch.split(x, (C, C, self.focal_level + 1), 1)

        ctx_all = 0
        for l in range(self.focal_level):
            ctx = self.focal_layers[l](ctx)
            ctx_all = ctx_all + ctx * gates[:, l:l + 1]
        ctx_global = self.act(ctx.mean(2, keepdim=True).mean(3, keepdim=True))
        ctx_all = ctx_all + ctx_global * gates[:, self.focal_level:]

        x_out = q * self.h(ctx_all)
        x_out = x_out.permute(0, 2, 3, 1).contiguous()
        if self.use_postln:
            x_out = self.ln(x_out)
        x_out = self.proj(x_out)
        x_out = self.proj_drop(x_out)
        return x_out
##########################################################################

####替换FeedForward######################################################################
class Mlp(nn.Module):
    """ Multilayer perceptron."""

    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0.):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x
class FocalModulationBlock(nn.Module):
    def __init__(self, dim, mlp_ratio=4., drop=0., drop_path=0.,
                 act_layer=nn.GELU, norm_layer=LayerNorm,
                 focal_level=2, focal_window=5, use_layerscale=True, layerscale_value=1e-4, LayerNorm_type='WithBias'):
        super().__init__()
        self.norm1 = norm_layer(dim, LayerNorm_type)
        self.modulation = FocalModulation(
            dim=dim,
            proj_drop=drop
        )
        self.drop_path = nn.Identity() if drop_path == 0. else DropPath(drop_path)
        self.norm2 = norm_layer(dim, LayerNorm_type)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(
            in_features=dim,
            hidden_features=mlp_hidden_dim,
            act_layer=act_layer,
            drop=drop
        )

        self.gamma_1 = 1.0
        self.gamma_2 = 1.0
        if use_layerscale:
            self.gamma_1 = nn.Parameter(layerscale_value * torch.ones((dim)), requires_grad=True)
            self.gamma_2 = nn.Parameter(layerscale_value * torch.ones((dim)), requires_grad=True)

    def forward(self, x):
        B, C, H, W = x.shape
        shortcut = x
        x = self.norm1(x)

        x = rearrange(x, 'b c h w -> b h w c')
        x = self.modulation(x)
        x = rearrange(x, 'b h w c -> b (h w) c')
        x = self.mlp(x)
        x = rearrange(x, 'b (h w) c -> b c h w', h=H, w=W)

        x = shortcut + self.drop_path(self.gamma_1 * x)

        shortcut = x
        x = self.norm2(x)

        x = rearrange(x, 'b c h w -> b (h w) c')
        x = self.mlp(x)
        x = rearrange(x, 'b (h w) c -> b c h w', h=H, w=W)

        x = shortcut + self.drop_path(self.gamma_2 * x)

        return x

##########################################################################

##########################################################################

##########################################################################
## Overlapped image patch embedding with 3x3 Conv
class OverlapPatchEmbed(nn.Module):
    def __init__(self, inp_channels, dim):
        super().__init__()
        self.proj = nn.Conv2d(inp_channels, dim, kernel_size=3, stride=1, padding=1)

    def forward(self, x):
        x = self.proj(x)
        return x



# 修改后的 Restormer_Encoder 类
class Restormer_Encoder(nn.Module):
    def __init__(self,
                 inp_channels=1,
                 out_channels=1,
                 dim=64,
                 num_blocks=[4, 4],
                 heads=[8, 8, 8],
                 ffn_expansion_factor=2,
                 bias=False,
                 LayerNorm_type='WithBias',
                 mlp_ratio=4.,
                 drop=0.,
                 drop_path=0.,
                 focal_level=2,
                 focal_window=5,
                 use_layerscale=False,
                 layerscale_value=1e-4,
                 proj_drop=0.
                 ):
        super().__init__()

        self.patch_embed = OverlapPatchEmbed(inp_channels, dim)

        self.encoder_level1 = nn.Sequential(
            *[FocalModulationBlock(
                dim=dim,
                mlp_ratio=mlp_ratio,
                drop=drop,
                drop_path=drop_path,
                act_layer=nn.GELU,
                norm_layer=LayerNorm,
                use_layerscale=use_layerscale,
                layerscale_value=layerscale_value,
                LayerNorm_type=LayerNorm_type
            ) for i in range(num_blocks[0])]
        )

        self.baseFeature = BaseFeatureExtraction(
            dim=dim,
            num_heads=heads[2],
            ffn_expansion_factor=ffn_expansion_factor,
            qkv_bias=bias,
            mlp_ratio=mlp_ratio,
            drop=drop,
            proj_drop=proj_drop,
            LayerNorm_type=LayerNorm_type
        )
        self.detailFeature = DetailFeatureExtraction()

    def forward(self, inp_img):
        inp_enc_level1 = self.patch_embed(inp_img)
        out_enc_level1 = self.encoder_level1(inp_enc_level1)

        base_feature = self.baseFeature(out_enc_level1)
        detail_feature = self.detailFeature(out_enc_level1)
        return base_feature, detail_feature, out_enc_level1





# 修改后的 Restormer_Decoder 类
class Restormer_Decoder(nn.Module):
    def __init__(self,
                 inp_channels=1,
                 out_channels=1,
                 dim=64,
                 num_blocks=[4, 4],
                 heads=[8, 8, 8],
                 ffn_expansion_factor=2,
                 bias=False,
                 LayerNorm_type='WithBias',
                 mlp_ratio=4.,
                 drop=0.,
                 drop_path=0.,
                 focal_level=2,
                 focal_window=5,
                 use_layerscale=False,
                 layerscale_value=1e-4,
                 proj_drop=0.
                 ):
        super().__init__()
        self.reduce_channel = nn.Conv2d(int(dim * 2), int(dim), kernel_size=1, bias=bias)
        self.encoder_level2 = nn.Sequential(
            *[FocalModulationBlock(
                dim=dim,
                mlp_ratio=mlp_ratio,
                drop=drop,
                drop_path=drop_path,
                act_layer=nn.GELU,
                norm_layer=LayerNorm,
                focal_level=focal_level,
                focal_window=focal_window,
                use_layerscale=use_layerscale,
                layerscale_value=layerscale_value,
                LayerNorm_type=LayerNorm_type
            ) for i in range(num_blocks[1])]
        )
        self.output = nn.Sequential(
            nn.Conv2d(int(dim), int(dim) // 2, kernel_size=3,
                      stride=1, padding=1, bias=bias),
            nn.LeakyReLU(),
            nn.Conv2d(int(dim) // 2, out_channels, kernel_size=3,
                      stride=1, padding=1, bias=bias),
        )
        self.sigmoid = nn.Sigmoid()

    def forward(self, inp_img, base_feature, detail_feature):
        out_enc_level0 = torch.cat((base_feature, detail_feature), dim=1)
        out_enc_level0 = self.reduce_channel(out_enc_level0)
        out_enc_level1 = self.encoder_level2(out_enc_level0)
        if inp_img is not None:
            out_enc_level1 = self.output(out_enc_level1) + inp_img
        else:
            out_enc_level1 = self.output(out_enc_level1)

        return self.sigmoid(out_enc_level1), out_enc_level0




if __name__ == "__main__":
    # 定义模型参数
    config = {
        "inp_channels": 3,
        "out_channels": 3,
        "dim": 64,
        "num_blocks": [4, 4],
        "heads": [8, 8, 8],
        "ffn_expansion_factor": 2,
        "bias": False,
        "LayerNorm_type": 'WithBias',
        "mlp_ratio": 4.,
        "drop": 0.,
        "drop_path": 0.,
        "focal_level": 2,
        "focal_window": 9,
        "use_layerscale": False,
        "layerscale_value": 1e-4,
        "proj_drop": 0.
    }

    # 初始化编码器和解码器
    encoder = Restormer_Encoder(**config)
    decoder = Restormer_Decoder(**config)

    # 生成随机输入图像
    batch_size = 2
    height = 64
    width = 64
    input_image = torch.randn(batch_size, config["inp_channels"], height, width)

    # 编码器前向传播
    base_feature, detail_feature, out_enc_level1 = encoder(input_image)

    # 检查编码器输出是否为张量
    assert isinstance(base_feature, torch.Tensor), "编码器输出的基础特征不是张量"
    assert isinstance(detail_feature, torch.Tensor), "编码器输出的细节特征不是张量"
    assert isinstance(out_enc_level1, torch.Tensor), "编码器输出的中间结果不是张量"

    # 解码器前向传播
    output_image, out_enc_level0 = decoder(input_image, base_feature, detail_feature)

    # 检查解码器输出是否为张量
    assert isinstance(output_image, torch.Tensor), "解码器输出的图像不是张量"
    assert isinstance(out_enc_level0, torch.Tensor), "解码器输出的中间结果不是张量"

    # 检查输出的批次大小是否与输入一致
    assert output_image.shape[0] == batch_size, "解码器输出的批次大小与输入不一致"

    print("所有测试用例通过！")
