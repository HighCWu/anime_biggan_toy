import numpy as np
import paddle.fluid as fluid
from paddle.fluid import layers, dygraph as dg
from paddle.fluid.initializer import Normal, Constant, Uniform


class ModelCache(object):
    G = None
    D = None
    train_mode = False
    initialized = False
model_cache = ModelCache


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
    value = layers.transpose(value, [0,2,3,1])
    sh = value.shape
    dim = len(sh[1:-1])
    out = (layers.reshape(value, [-1] + sh[-dim:]))
    for i in range(dim, 0, -1):
        out = layers.concat([out, layers.zeros_like(out)], i)
    out_size = [-1] + [s * 2 for s in sh[1:-1]] + [sh[-1]]
    out = layers.reshape(out, out_size)
    out = layers.transpose(out, [0,3,1,2])
    return out
 

class ReLU(dg.Layer):
    def forward(self, x):
        return layers.relu(x)
    
 
class SoftMax(dg.Layer):
    def __init__(self, **kwargs):
        super().__init__()
        self.kwargs = kwargs
  
    def forward(self, x):
        return layers.softmax(x, **self.kwargs)
 

class BatchNorm(dg.BatchNorm): # not trainable
    def __init__(self, *args, **kwargs):
        if 'affine' in kwargs:
            affine = kwargs.pop('affine')
        else:
            affine = True
        super().__init__(*args, **kwargs)
        self._use_global_stats = True
        if not affine:
            weight = (self.weight * 0 + 1).detach()
            bias = (self.bias * 0).detach()
            del self._parameters['bias']
            del self._parameters['weight']
            self.weight = weight
            self.bias = bias
        self.weight.stop_gradient = True
        self.bias.stop_gradient = True
        self.accumulated_mean = self.create_parameter(shape=[args[0]], default_initializer=Constant(0.0))
        self.accumulated_var = self.create_parameter(shape=[args[0]], default_initializer=Constant(0.0))
        self.accumulated_counter = self.create_parameter(shape=[1], default_initializer=Constant(1e-12))
        self.accumulated_mean.stop_gradient = True
        self.accumulated_var.stop_gradient = True
        self.accumulated_counter.stop_gradient = True
 
    def forward(self, inputs, *args, **kwargs):
        if '_mean' in self._parameters:
            del self._parameters['_mean']
        if '_variance' in self._parameters:
            del self._parameters['_variance']
        if not model_cache.initialized and not model_cache.train_mode:
            self._mean = (self.accumulated_mean / self.accumulated_counter)
            self._variance = (self.accumulated_var / self.accumulated_counter)
        if model_cache.train_mode:
            axes = [0] + ([] if len(inputs.shape) == 2 else list(range(2,len(inputs.shape))))
            _mean = layers.reduce_mean(inputs, axes, keep_dim=True)
            self._mean = layers.reduce_mean(inputs, axes, keep_dim=False)
            self._variance = layers.reduce_mean((inputs-_mean)**2, axes)
        else:
            self._mean = self._mean.detach()
            self._variance = self._variance.detach()
        return super().forward(inputs, *args, **kwargs)
 
 
class SpectralNorm(dg.Layer): # not trainable
    def __init__(self, module, name='weight', power_iterations=2):
        super().__init__()
        self.module = module
        self.name = name
        self.power_iterations = power_iterations
        self.initialized = False
        if not self._made_params():
            self._make_params()
 
    def _update_u(self):
        w = self.weight
        u = self.weight_u
 
        if len(w.shape) == 4:
            _w = layers.transpose(w, [2,3,1,0])
            _w = layers.reshape(_w, [-1, _w.shape[-1]])
        else:
            _w = layers.reshape(w, [-1, w.shape[-1]])
            _w = layers.reshape(_w, [-1, _w.shape[-1]])
        singular_value = "left" if _w.shape[0] <= _w.shape[1] else "right"
        norm_dim = 0 if _w.shape[0] <= _w.shape[1] else 1
        for _ in range(self.power_iterations):
            if singular_value == "left":
                v = layers.l2_normalize(layers.matmul(_w, u, transpose_x=True), axis=norm_dim)
                u = layers.l2_normalize(layers.matmul(_w, v), axis=norm_dim)
            else:
                v = layers.l2_normalize(layers.matmul(u, _w, transpose_y=True), axis=norm_dim)
                u = layers.l2_normalize(layers.matmul(v, _w), axis=norm_dim)
 
        if singular_value == "left":
            sigma = layers.matmul(layers.matmul(u, _w, transpose_x=True), v)
        else:
            sigma = layers.matmul(layers.matmul(v, _w), u, transpose_y=True)
        _w = w / sigma.detach()
        setattr(self.module, self.name, _w.detach()) # setattr(self.module, self.name, _w)
        # self.weight_u.set_value(u)
 
    def _made_params(self):
        try:
            self.weight
            self.weight_u
            return True
        except AttributeError:
            return False
 
    def _make_params(self):
        # paddle linear weight is similar with tf's, and conv weight is similar with pytorch's.
        w = getattr(self.module, self.name)
 
        if len(w.shape) == 4:
            _w = layers.transpose(w, [2,3,1,0])
            _w = layers.reshape(_w, [-1, _w.shape[-1]]) 
        else:
            _w = layers.reshape(w, [-1, w.shape[-1]])
        singular_value = "left" if _w.shape[0] <= _w.shape[1] else "right"
        norm_dim = 0 if _w.shape[0] <= _w.shape[1] else 1
        u_shape = (_w.shape[0], 1) if singular_value == "left" else (1, _w.shape[-1])
        
        u = self.create_parameter(shape=u_shape, default_initializer=Normal(0, 1))
        u.stop_gradient = True
        u.set_value(layers.l2_normalize(u, axis=norm_dim))
 
        del self.module._parameters[self.name]
        self.add_parameter("weight", w)
        self.add_parameter("weight_u", u)
 
    def forward(self, *args, **kwargs):
        if not self.initialized:
            self._update_u()
            self.initialized = True
        return self.module.forward(*args, **kwargs)
    
    
class SelfAttention(dg.Layer):
    def __init__(self, in_dim, activation=layers.relu):
        super().__init__()
        self.chanel_in = in_dim
        self.activation = activation
    
        self.theta = SpectralNorm(dg.Conv2D(in_dim, in_dim // 8, 1, bias_attr=False))
        self.phi = SpectralNorm(dg.Conv2D(in_dim, in_dim // 8, 1, bias_attr=False))
        self.pool = dg.Pool2D(2, 'max', 2)
        self.g = SpectralNorm(dg.Conv2D(in_dim, in_dim // 2, 1, bias_attr=False))
        self.o_conv = SpectralNorm(dg.Conv2D(in_dim // 2, in_dim, 1, bias_attr=False))
        self.gamma = self.create_parameter([1,], default_initializer=Constant(0.0))
    
        self.softmax = SoftMax(axis=-1)
    
    def forward(self, x):
        m_batchsize, C, width, height = x.shape
        N = height * width
    
        theta = self.theta(x)
        phi = self.phi(x)
        phi = self.pool(phi)
        phi = layers.reshape(phi,(m_batchsize, -1, N // 4))
        theta = layers.reshape(theta,(m_batchsize, -1, N))
        theta = layers.transpose(theta,(0, 2, 1))
        attention = self.softmax(layers.bmm(theta, phi))
        g = self.g(x)
        g = layers.reshape(self.pool(g),(m_batchsize, -1, N // 4))
        attn_g = layers.reshape(layers.bmm(g, layers.transpose(attention,(0, 2, 1))),(m_batchsize, -1, width, height))
        out = self.o_conv(attn_g)
        return self.gamma * out + x
 
 
class ConditionalBatchNorm(dg.Layer):
    def __init__(self, num_features, num_classes, epsilon=1e-5, momentum=0.1):
        super().__init__()
        self.bn_in_cond = BatchNorm(num_features, affine=False, epsilon=epsilon, momentum=momentum)
        self.gamma_embed = SpectralNorm(dg.Linear(num_classes, num_features, bias_attr=False))
        self.beta_embed = SpectralNorm(dg.Linear(num_classes, num_features, bias_attr=False))
    
    def forward(self, x, y):
        out = self.bn_in_cond(x)
        if isinstance(y, list):
            gamma, beta = y
            out = layers.reshape(gamma, (0, 0, 1, 1)) * out + layers.reshape(beta, (0, 0, 1, 1))
            return out

        gamma = self.gamma_embed(y)
        beta = self.beta_embed(y)
        out = layers.reshape(gamma, (0, 0, 1, 1)) * out + layers.reshape(beta, (0, 0, 1, 1))
        return out
 
 
class ResBlock(dg.Layer):
    def __init__(
        self,
        in_channel,
        out_channel,
        kernel_size=[3, 3],
        padding=1,
        stride=1,
        n_class=None,
        conditional=True,
        activation=layers.relu,
        upsample=True,
        downsample=False,
        z_dim=128,
        use_attention=False,
        skip_proj=None
    ):
        super().__init__()
    
        if conditional:
            self.cond_norm1 = ConditionalBatchNorm(in_channel, z_dim)
    
        self.conv0 = SpectralNorm(
            dg.Conv2D(in_channel, out_channel, kernel_size, stride, padding)
        )
    
        if conditional:
            self.cond_norm2 = ConditionalBatchNorm(out_channel, z_dim)
    
        self.conv1 = SpectralNorm(
            dg.Conv2D(out_channel, out_channel, kernel_size, stride, padding)
        )
    
        self.skip_proj = False
        if skip_proj is not True and (upsample or downsample):
            self.conv_sc = SpectralNorm(dg.Conv2D(in_channel, out_channel, 1, 1, 0))
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
            out = self.cond_norm1(out, condition[0] if isinstance(condition, list) else condition)
        out = self.activation(out)
        if self.upsample:
            out = unpool(out)
        out = self.conv0(out)
        
        if self.conditional:
            out = self.cond_norm2(out, condition[1] if isinstance(condition, list) else condition)
        out = self.activation(out)
        out = self.conv1(out)
    
        if self.downsample:
            out = layers.pool2d(out, 2, pool_type='avg', pool_stride=2)
    
        if self.skip_proj:
            skip = input
            if self.upsample:
                skip = unpool(skip)
            skip = self.conv_sc(skip)
            if self.downsample:
                skip = layers.pool2d(skip, 2, pool_type='avg', pool_stride=2)
            out = out + skip
        else:
            skip = input
    
        if self.use_attention:
            out = self.attention(out)
    
        return out
 
 
class Generator(dg.Layer): # not trainable
    def __init__(self, code_dim=128, n_class=1000, chn=96, blocks_with_attention="B4", resolution=512):
        super().__init__()
    
        def GBlock(in_channel, out_channel, n_class, z_dim, use_attention):
            return ResBlock(in_channel, out_channel, n_class=n_class, z_dim=z_dim, use_attention=use_attention)
    
        self.embed_y = dg.Linear(n_class, 128, bias_attr=False)
    
        self.chn = chn
        self.resolution = resolution 
        self.blocks_with_attention = set(blocks_with_attention.split(",")) 
        self.blocks_with_attention.discard('')
    
        gblock = []
        in_channels, out_channels = self.get_in_out_channels()
        self.num_split = len(in_channels) + 1
    
        z_dim = code_dim//self.num_split + 128
        self.noise_fc = SpectralNorm(dg.Linear(code_dim//self.num_split, 4 * 4 * in_channels[0]))
    
        self.sa_ids = [int(s.split('B')[-1]) for s in self.blocks_with_attention]
    
        for i, (nc_in, nc_out) in enumerate(zip(in_channels, out_channels)):
            gblock.append(GBlock(nc_in, nc_out, n_class=n_class, z_dim=z_dim, use_attention=(i+1) in self.sa_ids))
        self.blocks = dg.LayerList(gblock)
    
        self.output_layer_bn = BatchNorm(1 * chn, epsilon=1e-5)
        self.output_layer_conv = SpectralNorm(dg.Conv2D(1 * chn, 3, [3, 3], padding=1))
 
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

    def forward(self, input, class_id, input_class_emb=False):
        if isinstance(input, list):
            codes = [input[0]]
            codes += [input[2*i+1:2*i+3] for i in range(len(input)//2)]
        else:
            codes = layers.split(input, self.num_split, 1)
        if not input_class_emb:
            class_emb = self.embed_y(class_id)  # 128
        else:
            class_emb = class_id
        out = self.noise_fc(codes[0])
        out = layers.transpose(layers.reshape(out,(out.shape[0], 4, 4, -1)),(0, 3, 1, 2))
        for i, (code, gblock) in enumerate(zip(codes[1:], self.blocks)):
            if isinstance(input, list):
                condition = [layers.concat([c, class_emb], 1) for c in code]
            else:
                condition = layers.concat([code, class_emb], 1)
            out = gblock(out, condition)
    
        out = self.output_layer_bn(out)
        out = layers.relu(out)
        out = self.output_layer_conv(out)
 
        return (layers.tanh(out) + 1) / 2
        
 
class Discriminator(dg.Layer):
    def __init__(self, n_class=1000, chn=96, blocks_with_attention="B2", resolution=256): 
        super().__init__()
    
        def DBlock(in_channel, out_channel, downsample=True, use_attention=False, skip_proj=None):
            return ResBlock(in_channel, out_channel, conditional=False, upsample=False, 
                        downsample=downsample, use_attention=use_attention, skip_proj=skip_proj)
    
        self.chn = chn
        self.colors = 3
        self.resolution = resolution  
        self.blocks_with_attention = set(blocks_with_attention.split(",")) 
        self.blocks_with_attention.discard('')
    
        dblock = []
        in_channels, out_channels = self.get_in_out_channels()
    
        self.sa_ids = [int(s.split('B')[-1]) for s in self.blocks_with_attention]
    
        for i, (nc_in, nc_out) in enumerate(zip(in_channels[:-1], out_channels[:-1])):
            dblock.append(DBlock(nc_in, nc_out, downsample=True, 
                        use_attention=(i+1) in self.sa_ids, skip_proj=nc_in==nc_out))
        dblock.append(DBlock(in_channels[-1], out_channels[-1], downsample=False, 
                        use_attention=len(out_channels) in self.sa_ids, skip_proj=in_channels[-1]==out_channels[-1]))
        self.blocks = dg.LayerList(dblock)
    
        self.final_fc = SpectralNorm(dg.Linear(16 * chn, 1))
    
        self.embed_y = dg.Embedding(size=[n_class, 16 * chn], is_sparse=False, param_attr=Uniform(-0.1,0.1))
        self.embed_y = SpectralNorm(self.embed_y)
 
    def get_in_out_channels(self):
        colors = self.colors
        resolution = self.resolution
        if resolution == 1024:
            channel_multipliers = [1, 1, 1, 2, 4, 8, 8, 16, 16]
        elif resolution == 512:
            channel_multipliers = [1, 1, 2, 4, 8, 8, 16, 16]
        elif resolution == 256:
            channel_multipliers = [1, 2, 4, 8, 8, 16, 16]
        elif resolution == 128:
            channel_multipliers = [1, 2, 4, 8, 16, 16]
        elif resolution == 64:
            channel_multipliers = [2, 4, 8, 16, 16]
        elif resolution == 32:
            channel_multipliers = [2, 2, 2, 2]
        else:
            raise ValueError("Unsupported resolution: {}".format(resolution))
        out_channels = [self.chn * c for c in channel_multipliers]
        in_channels = [colors] + out_channels[:-1]
        return in_channels, out_channels
 
    def forward(self, input, class_id=None):
        out = input
        features = []
        for i, dblock in enumerate(self.blocks):
            out = dblock(out)
            features.append(out)
        out = layers.relu(out)
        out = layers.reduce_sum(out, [2,3])
        out_linear = self.final_fc(out)
        if class_id is None:
            prod = 0
        else:
            class_emb = self.embed_y(class_id) 
        
            prod = layers.reduce_sum((class_emb * out), 1, keep_dim=True)
        
        return layers.sigmoid(out_linear + prod), features
