#@title Define Generator and Discriminator model
import numpy as np
import torch
from torch import nn
from torch.nn import Parameter
from torch.nn import functional as F


def l2_normalize(v, dim=None, eps=1e-12):
    return v / (v.norm(dim=dim, keepdim=True) + eps)
    
 
def unpool(value):
    """Unpooling operation.
    N-dimensional version of the unpooling operation from
    https://www.robots.ox.ac.uk/~vgg/rg/papers/Dosovitskiy_Learning_to_Generate_2015_CVPR_paper.pdf
    Taken from: https://github.com/tensorflow/tensorflow/issues/2169
    Args:
        value: a Tensor of shape [b, d0, d1, ..., dn, ch]
        name: name of the op
    Returns:
        A Tensor of shape [b, 2*d0, 2*d1, ..., 2*dn, ch]
    """
    value = torch.Tensor.permute(value, [0,2,3,1])
    sh = list(value.shape)
    dim = len(sh[1:-1])
    out = (torch.reshape(value, [-1] + sh[-dim:]))
    for i in range(dim, 0, -1):
        out = torch.cat([out, torch.zeros_like(out)], i)
    out_size = [-1] + [s * 2 for s in sh[1:-1]] + [sh[-1]]
    out = torch.reshape(out, out_size)
    out = torch.Tensor.permute(out, [0,3,1,2])
    return out
 
 
class BatchNorm2d(nn.BatchNorm2d):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.initialized = False
        self.accumulating = False
        self.accumulated_mean = Parameter(torch.zeros(args[0]), requires_grad=False)
        self.accumulated_var = Parameter(torch.zeros(args[0]), requires_grad=False)
        self.accumulated_counter = Parameter(torch.zeros(1)+1e-12, requires_grad=False)
 
    def forward(self, inputs, *args, **kwargs):
        if not self.initialized:
            self.check_accumulation()
            self.set_initialized(True)
        if self.accumulating:
            self.eval()
            with torch.no_grad():
                axes = [0] + ([] if len(inputs.shape) == 2 else list(range(2,len(inputs.shape))))
                _mean = torch.mean(inputs, axes, keepdim=True)
                mean = torch.mean(inputs, axes, keepdim=False)
                var = torch.mean((inputs-_mean)**2, axes)
                self.accumulated_mean.copy_(self.accumulated_mean + mean)
                self.accumulated_var.copy_(self.accumulated_var + var)
                self.accumulated_counter.copy_(self.accumulated_counter + 1)
                _mean = self.running_mean*1.0
                _variance = self.running_var*1.0
                self._mean.copy_(self.accumulated_mean / self.accumulated_counter)
                self._variance.copy_(self.accumulated_var / self.accumulated_counter)
                out = super().forward(inputs, *args, **kwargs)
                self.running_mean.copy_(_mean)
                self.running_var.copy_(_variance)
                return out
        out = super().forward(inputs, *args, **kwargs)
        return out
 
    def check_accumulation(self):
        if self.accumulated_counter.detach().cpu().numpy().mean() > 1-1e-12:
            self.running_mean.copy_(self.accumulated_mean / self.accumulated_counter)
            self.running_var.copy_(self.accumulated_var / self.accumulated_counter)
            return True
        return False
 
    def clear_accumulated(self):
        self.accumulated_mean.copy_(self.accumulated_mean*0.0)
        self.accumulated_var.copy_(self.accumulated_var*0.0)
        self.accumulated_counter.copy_(self.accumulated_counter*0.0+1e-2)
 
    def set_accumulating(self, status=True):
        if status:
            self.accumulating = True
        else:
            self.accumulating = False
 
    def set_initialized(self, status=False):
        if not status:
            self.initialized = False
        else:
            self.initialized = True
 

class SpectralNorm(nn.Module):
    def __init__(self, module, name='weight', power_iterations=2):
        super().__init__()
        self.module = module
        self.name = name
        self.power_iterations = power_iterations
        if not self._made_params():
            self._make_params()
 
    def _update_u(self):
        w = self.weight
        u = self.weight_u
 
        if len(w.shape) == 4:
            _w = torch.Tensor.permute(w, [2,3,1,0])
            _w = torch.reshape(_w, [-1, _w.shape[-1]])
        elif isinstance(self.module, nn.Linear) or isinstance(self.module, nn.Embedding):
            _w = torch.Tensor.permute(w, [1,0])
            _w = torch.reshape(_w, [-1, _w.shape[-1]])
        else:
            _w = torch.reshape(w, [-1, w.shape[-1]])
            _w = torch.reshape(_w, [-1, _w.shape[-1]])
        singular_value = "left" if _w.shape[0] <= _w.shape[1] else "right"
        norm_dim = 0 if _w.shape[0] <= _w.shape[1] else 1
        for _ in range(self.power_iterations):
            if singular_value == "left":
                v = l2_normalize(torch.matmul(_w.t(), u), dim=norm_dim)
                u = l2_normalize(torch.matmul(_w, v), dim=norm_dim)
            else:
                v = l2_normalize(torch.matmul(u, _w.t()), dim=norm_dim)
                u = l2_normalize(torch.matmul(v, _w), dim=norm_dim)
 
        if singular_value == "left":
            sigma = torch.matmul(torch.matmul(u.t(), _w), v)
        else:
            sigma = torch.matmul(torch.matmul(v, _w), u.t())
        _w = w / sigma.detach()
        setattr(self.module, self.name, _w)
        self.weight_u.copy_(u.detach())
 
    def _made_params(self):
        try:
            self.weight
            self.weight_u
            return True
        except AttributeError:
            return False
 
    def _make_params(self):
        w = getattr(self.module, self.name)
 
        if len(w.shape) == 4:
            _w = torch.Tensor.permute(w, [2,3,1,0])
            _w = torch.reshape(_w, [-1, _w.shape[-1]])
        elif isinstance(self.module, nn.Linear) or isinstance(self.module, nn.Embedding):
            _w = torch.Tensor.permute(w, [1,0])
            _w = torch.reshape(_w, [-1, _w.shape[-1]])
        else:
            _w = torch.reshape(w, [-1, w.shape[-1]])
        singular_value = "left" if _w.shape[0] <= _w.shape[1] else "right"
        norm_dim = 0 if _w.shape[0] <= _w.shape[1] else 1
        u_shape = (_w.shape[0], 1) if singular_value == "left" else (1, _w.shape[-1])
        
        u = Parameter(w.data.new(*u_shape).normal_(0, 1), requires_grad=False)
        u.copy_(l2_normalize(u, dim=norm_dim).detach())
 
        del self.module._parameters[self.name]
        self.weight = w
        self.weight_u = u
 
    def forward(self, *args, **kwargs):
        self._update_u()
        return self.module.forward(*args, **kwargs)
    
    
class SelfAttention(nn.Module):
    def __init__(self, in_dim, activation=torch.relu):
        super().__init__()
        self.chanel_in = in_dim
        self.activation = activation
    
        self.theta = SpectralNorm(nn.Conv2d(in_dim, in_dim // 8, 1, bias=False))
        self.phi = SpectralNorm(nn.Conv2d(in_dim, in_dim // 8, 1, bias=False))
        self.pool = nn.MaxPool2d(2, 2)
        self.g = SpectralNorm(nn.Conv2d(in_dim, in_dim // 2, 1, bias=False))
        self.o_conv = SpectralNorm(nn.Conv2d(in_dim // 2, in_dim, 1, bias=False))
        self.gamma = Parameter(torch.zeros(1))
    
    def forward(self, x):
        m_batchsize, C, width, height = x.shape
        N = height * width
    
        theta = self.theta(x)
        phi = self.phi(x)
        phi = self.pool(phi)
        phi = torch.reshape(phi,(m_batchsize, -1, N // 4))
        theta = torch.reshape(theta,(m_batchsize, -1, N))
        theta = torch.Tensor.permute(theta,(0, 2, 1))
        attention = torch.softmax(torch.bmm(theta, phi), -1)
        g = self.g(x)
        g = torch.reshape(self.pool(g),(m_batchsize, -1, N // 4))
        attn_g = torch.reshape(torch.bmm(g, torch.Tensor.permute(attention,(0, 2, 1))),(m_batchsize, -1, width, height))
        out = self.o_conv(attn_g)
        return self.gamma * out + x
 
 
class ConditionalBatchNorm2d(nn.Module):
    def __init__(self, num_features, num_classes, eps=1e-5, momentum=0.1):
        super().__init__()
        self.bn_in_cond = BatchNorm2d(num_features, affine=False, eps=eps, momentum=momentum)
        self.gamma_embed = SpectralNorm(nn.Linear(num_classes, num_features, bias=False))
        self.beta_embed = SpectralNorm(nn.Linear(num_classes, num_features, bias=False))
    
    def forward(self, x, y):
        out = self.bn_in_cond(x)

        if isinstance(y, list):
            gamma, beta = y
            out = torch.reshape(gamma, (gamma.shape[0], -1, 1, 1)) * out + torch.reshape(beta, (beta.shape[0], -1, 1, 1))
            return out

        gamma = self.gamma_embed(y)
        # gamma = gamma + 1
        beta = self.beta_embed(y)
        out = torch.reshape(gamma, (gamma.shape[0], -1, 1, 1)) * out + torch.reshape(beta, (beta.shape[0], -1, 1, 1))
        return out
 

class ResBlock(nn.Module):
    def __init__(
        self,
        in_channel,
        out_channel,
        kernel_size=[3, 3],
        padding=1,
        stride=1,
        n_class=None,
        conditional=True,
        activation=torch.relu,
        upsample=True,
        downsample=False,
        z_dim=128,
        use_attention=False,
        skip_proj=None
    ):
        super().__init__()
    
        if conditional:
            self.cond_norm1 = ConditionalBatchNorm2d(in_channel, z_dim)
    
        self.conv0 = SpectralNorm(
            nn.Conv2d(in_channel, out_channel, kernel_size, stride, padding)
        )
    
        if conditional:
            self.cond_norm2 = ConditionalBatchNorm2d(out_channel, z_dim)
    
        self.conv1 = SpectralNorm(
            nn.Conv2d(out_channel, out_channel, kernel_size, stride, padding)
        )
    
        self.skip_proj = False
        if skip_proj is not True and (upsample or downsample):
            self.conv_sc = SpectralNorm(nn.Conv2d(in_channel, out_channel, 1, 1, 0))
            self.skip_proj = True
    
        if use_attention:
            self.attention = SelfAttention(out_channel)
    
        self.upsample = upsample
        self.downsample = downsample
        self.activation = activation
        self.conditional = conditional
        self.use_attention = use_attention
    
    def forward(self, input, condition=None):
        out = input
    
        if self.conditional:
            out = self.cond_norm1(out, condition if not isinstance(condition, list) else condition[0])
        out = self.activation(out)
        if self.upsample:
            out = unpool(out) # out = F.interpolate(out, scale_factor=2)
        out = self.conv0(out)
        if self.conditional:
            out = self.cond_norm2(out, condition if not isinstance(condition, list) else condition[1])
        out = self.activation(out)
        out = self.conv1(out)
        
        if self.downsample:
            out = F.avg_pool2d(out, 2, 2)
    
        if self.skip_proj:
            skip = input
            if self.upsample:
                skip = unpool(skip) # skip = F.interpolate(skip, scale_factor=2)
            skip = self.conv_sc(skip)
            if self.downsample:
                skip = F.avg_pool2d(skip, 2, 2)
            out = out + skip
        else:
            skip = input
    
        if self.use_attention:
            out = self.attention(out)
    
        return out
 
 
class Generator(nn.Module):
    def __init__(self, code_dim=128, n_class=1000, chn=96, blocks_with_attention="B4", resolution=512):
        super().__init__()
    
        def GBlock(in_channel, out_channel, n_class, z_dim, use_attention):
            return ResBlock(in_channel, out_channel, n_class=n_class, z_dim=z_dim, use_attention=use_attention)
    
        self.embed_y = nn.Linear(n_class, 128, bias=False)
    
        self.chn = chn
        self.resolution = resolution 
        self.blocks_with_attention = set(blocks_with_attention.split(",")) 
        self.blocks_with_attention.discard('')
    
        gblock = []
        in_channels, out_channels = self.get_in_out_channels()
        self.num_split = len(in_channels) + 1
    
        z_dim = code_dim//self.num_split + 128
        self.noise_fc = SpectralNorm(nn.Linear(code_dim//self.num_split, 4 * 4 * in_channels[0]))
    
        self.sa_ids = [int(s.split('B')[-1]) for s in self.blocks_with_attention]
    
        for i, (nc_in, nc_out) in enumerate(zip(in_channels, out_channels)):
            gblock.append(GBlock(nc_in, nc_out, n_class=n_class, z_dim=z_dim, use_attention=(i+1) in self.sa_ids))
        self.blocks = nn.ModuleList(gblock)
    
        self.output_layer_bn = BatchNorm2d(1 * chn, eps=1e-5)
        self.output_layer_conv = SpectralNorm(nn.Conv2d(1 * chn, 3, [3, 3], padding=1))

        self.z_dim = code_dim
        self.c_dim = n_class
        self.n_level = self.num_split
 
    def get_in_out_channels(self):
        resolution = self.resolution
        if resolution == 1024:
            channel_multipliers = [16, 16, 8, 8, 4, 2, 1, 1, 1]
        elif resolution == 512:
            channel_multipliers = [16, 16, 8, 8, 4, 2, 1, 1]
        elif resolution == 256:
            channel_multipliers = [16, 16, 8, 8, 4, 2, 1]
        elif resolution == 128:
            channel_multipliers = [16, 16, 8, 4, 2, 1]
        elif resolution == 64:
            channel_multipliers = [16, 16, 8, 4, 2]
        elif resolution == 32:
            channel_multipliers = [4, 4, 4, 4]
        else:
            raise ValueError("Unsupported resolution: {}".format(resolution))
        in_channels = [self.chn * c for c in channel_multipliers[:-1]]
        out_channels = [self.chn * c for c in channel_multipliers[1:]]
        return in_channels, out_channels
 
    def forward(self, input, class_id):
        codes = torch.chunk(input, self.num_split, 1)
        class_emb = self.embed_y(class_id)  # 128
        out = self.noise_fc(codes[0])
        out = torch.Tensor.permute(torch.reshape(out,(out.shape[0], 4, 4, -1)),(0, 3, 1, 2))
        for i, (code, gblock) in enumerate(zip(codes[1:], self.blocks)):
            condition = torch.cat([code, class_emb], 1)
            out = gblock(out, condition)
    
        out = self.output_layer_bn(out)
        out = torch.relu(out)
        out = self.output_layer_conv(out)
        
        return (torch.tanh(out) + 1) / 2

    def forward_w(self, ws):
        out = self.noise_fc(ws[0])
        out = torch.Tensor.permute(torch.reshape(out,(out.shape[0], 4, 4, -1)),(0, 3, 1, 2))
        for i, (w, gblock) in enumerate(zip(ws[1:], self.blocks)):
            out = gblock(out, w)
    
        out = self.output_layer_bn(out)
        out = torch.relu(out)
        out = self.output_layer_conv(out)
        
        return (torch.tanh(out) + 1) / 2

    def forward_wp(self, z0, gammas, betas):
        out = self.noise_fc(z0)
        out = torch.Tensor.permute(torch.reshape(out,(out.shape[0], 4, 4, -1)),(0, 3, 1, 2))
        for i, (gamma, beta, gblock) in enumerate(zip(gammas, betas, self.blocks)):
            out = gblock(out, [[gamma[0], beta[0]], [gamma[1], beta[1]]])
    
        out = self.output_layer_bn(out)
        out = torch.relu(out)
        out = self.output_layer_conv(out)
        
        return (torch.tanh(out) + 1) / 2
