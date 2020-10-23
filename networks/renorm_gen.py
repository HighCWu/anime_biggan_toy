import numpy as np
import paddle.fluid as fluid
from paddle.fluid import layers, dygraph as dg
from PIL import Image
from tqdm import tqdm
from .model import model_cache

from sys import stdout

class RandomState(object):
    rng = None
rds = RandomState

def renorm_gen(batch_size=8, seed=None):
    with dg.no_grad():
        model_cache.train_mode = True
        model_cache.initialized = True
        if seed is not None:
            rds.rng = np.random.RandomState(seed)
        elif rds.rng is None:
            rds.rng = np.random
        G = model_cache.G
        x_np = rds.rng.randn(batch_size,140).astype('float32')
        y_np = rds.rng.randint(0,1000,size=[batch_size]).astype('int64')
        x = dg.to_variable(x_np)
        y = dg.to_variable(y_np)
        y_hot = layers.one_hot(layers.unsqueeze(y,[1]), depth=1000)
        img_pd = G(x, y_hot)
        img = np.uint8(img_pd.numpy().clip(0,1)*255)
        imgs = []
        for i in range(len(img)):
            imgs += [Image.fromarray(img[i].transpose([1,2,0]))]
        return imgs

def renorm_gen_interpolate(batch_size=8, seed=None, out_path='data/out.gif'):
    with dg.no_grad():
        model_cache.train_mode = True
        model_cache.initialized = True
        if seed is not None:
            rds.rng = np.random.RandomState(seed)
        elif rds.rng is None:
            rds.rng = np.random
        G = model_cache.G
        x_np = rds.rng.randn(batch_size,140).astype('float32')
        y_np = rds.rng.randint(0,1000,size=[batch_size]).astype('int64')
        x = dg.to_variable(x_np)
        y = dg.to_variable(y_np)
        y_hot = layers.one_hot(layers.unsqueeze(y,[1]), depth=1000)
        y_embed = G.embed_y(y_hot)
        G(x, y_embed, True)
        model_cache.train_mode = False
        model_cache.initialized = True
        x = layers.concat([x, x[:1]], 0)
        y_embed = layers.concat([y_embed, y_embed[:1]], 0)
        imgs = []
        for i in range(batch_size):
            for j in range(40):
                alpha = j / (40 - 1)
                _x = (1 - alpha) * x[i:i+1] + alpha * x[i+1:i+2]
                _y_embed = (1 - alpha) *  y_embed[i:i+1] + alpha *  y_embed[i+1:i+2]
                img_pd = G(_x, _y_embed, True)
                img = np.uint8(img_pd.numpy().clip(0,1)*255)[0].transpose([1,2,0])
                imgs.append(Image.fromarray(img))
                stdout.write(f'{i*40+j+1}/{40*batch_size}\r')
                stdout.flush()
        print('')
        imgs[0].save(out_path, save_all=True, append_images=imgs[1:], duration=40, loop=0)
        return Image.open(out_path)
