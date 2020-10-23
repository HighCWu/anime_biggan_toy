import numpy as np
import paddle.fluid as fluid
from paddle.fluid import layers, dygraph as dg
from PIL import Image
from tqdm import tqdm
from .model import model_cache

from sys import stdout


class Latents(dg.Layer):
    def __init__(self):
        super(Latents, self).__init__()
        self.levels = ("y;z0;z11;z12;z21;z22;z31;z32;z41;z42;z51;z52;z61;z62").split(';')
        for level in self.levels:
            if len(level) == 1:
                self.add_parameter(level, self.create_parameter([1, 128]))
            if len(level) >= 2:
                self.add_parameter(level, self.create_parameter([1, 20]))

    def forward(self):
        z = [getattr(self, level) for level in self.levels[1:]]
        class_emb = getattr(self, self.levels[0])
        return z, class_emb


def std_enc(path='miku.png', steps=2000, lr=4e-3):
    model_cache.train_mode = False
    model_cache.initialized = False
    img = Image.open(path)
    w, h = img.size
    min_size = min(w, h)
    x0 = (w - min_size) // 2
    y0 = (h - min_size) // 2
    x1 = x0 + min_size
    y1 = y0 + min_size
    img = img.crop([x0,y0,x1,y1]).convert('RGB')
    img = _img = img.resize([256,256], Image.BILINEAR)
    img = np.asarray(img) / 255.0
    img = dg.to_variable(img.transpose(2,0,1).astype('float32')[None,...])
    m_latent = Latents()
    optimizer = fluid.optimizer.AdamOptimizer(learning_rate=lr, parameter_list=m_latent.parameters())
    for i in range(steps):
        z, class_emb = m_latent()
        out = model_cache.G(z, class_emb, input_class_emb=True)
        loss = layers.mean((out - img)**2)
        loss.backward()
        optimizer.minimize(loss)
        optimizer.clear_gradients()
        stdout.write(f'loss: {loss.numpy().mean()}  {i+1}/{steps}\r')
        stdout.flush()
    print('')
    out = np.uint8(out.numpy()[0].transpose(1,2,0).clip(0,1) * 255)
    return Image.fromarray(out), _img


def std_enc_with_D(path='miku.png', steps=2000, lr=4e-3, levels=[0,3], weights=[100,1]):
    model_cache.train_mode = False
    model_cache.initialized = False
    img = Image.open(path)
    w, h = img.size
    min_size = min(w, h)
    x0 = (w - min_size) // 2
    y0 = (h - min_size) // 2
    x1 = x0 + min_size
    y1 = y0 + min_size
    img = img.crop([x0,y0,x1,y1]).convert('RGB')
    img = _img = img.resize([256,256], Image.BILINEAR)
    img = np.asarray(img) / 255.0
    img = dg.to_variable(img.transpose(2,0,1).astype('float32')[None,...])
    m_latent = Latents()
    optimizer = fluid.optimizer.AdamOptimizer(learning_rate=lr, parameter_list=m_latent.parameters())
    for i in range(steps):
        z, class_emb = m_latent()
        out = model_cache.G(z, class_emb, input_class_emb=True)
        with dg.no_grad():
            _, real_features = model_cache.D(img)
            real_features = [img] + real_features
        _, fake_features = model_cache.D(out)
        fake_features = [out] + fake_features
        loss = 0
        for idx, weight in zip(levels, weights):
            r, f = real_features[idx], fake_features[idx]
            loss = loss + weight * layers.mean((f - r)**2)
        loss.backward()
        optimizer.minimize(loss)
        optimizer.clear_gradients()
        stdout.write(f'loss: {loss.numpy().mean()}  {i+1}/{steps}\r')
        stdout.flush()
    print('')
    out = np.uint8(out.numpy()[0].transpose(1,2,0).clip(0,1) * 255)
    return Image.fromarray(out), _img