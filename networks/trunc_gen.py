import os
import numpy as np
import paddle.fluid as fluid
from paddle.fluid import layers, dygraph as dg
from PIL import Image
from tqdm import tqdm
from scipy.stats import truncnorm
from .model import model_cache

from sys import stdout

class RandomState(object):
    rng = None
rds = RandomState

def trunc_gen(batch_size=8, trunc_range=(-1,1), truncation=0.5, seed=None):
    with dg.no_grad():
        model_cache.train_mode = False
        model_cache.initialized = False
        if seed is not None:
            rds.rng = np.random.RandomState(seed)
        elif rds.rng is None:
            rds.rng = np.random
        G = model_cache.G
        x_np = truncation * truncnorm.rvs(*trunc_range, size=(batch_size, 140), random_state=None if rds.rng is None else rds.rng).astype('float32')
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

def trunc_gen_interpolate(batch_size=8, trunc_range=(-2,2), truncation=1.0, seed=None, out_path='data/out',
                        levels=None, interpolate_mode=0):
    default_levels = ("y;z0;z11;z12;z21;z22;z31;z32;z41;z42;z51;z52;z61;z62")
    if levels is None:
        levels = default_levels
    default_levels = default_levels.split(';')

    img_save_dir = os.path.join('/tmp', out_path+'.dir')
    os.system(f'rm -rf {img_save_dir}')
    os.system(f'mkdir {img_save_dir} -p')

    with dg.no_grad():
        model_cache.train_mode = False
        model_cache.initialized = False
        if seed is not None:
            rds.rng = np.random.RandomState(seed)
        elif rds.rng is None:
            rds.rng = np.random
        G = model_cache.G
        x_np = truncation * truncnorm.rvs(*trunc_range, size=(batch_size, 140), random_state=None if rds.rng is None else rds.rng).astype('float32')
        y_np = rds.rng.randint(0,1000,size=[batch_size]).astype('int64')
        x = dg.to_variable(x_np)
        y_cls = dg.to_variable(y_np)
        y_hot = layers.one_hot(layers.unsqueeze(y_cls,[1]), depth=1000)
        y_embed = G.embed_y(y_hot)
        x = layers.concat([x, x[:1]], 0)
        y_embed = layers.concat([y_embed, y_embed[:1]], 0)
        levels = levels.split(';')
        for level in default_levels:
            if len(level) == 1:
                locals()[level] = y_embed
                locals()['_'+level] = y_embed[:1]
            if len(level) >= 2:
                idx = int(level[1])*20
                locals()[level] = x[:,idx:idx+20]
                locals()['_'+level] = x[:1,idx:idx+20]
        imgs = []
        for i in range(batch_size):
            for j in range(40):
                alpha = j / 40
                if interpolate_mode == 1:
                    alpha = alpha**2 * (3 - 2 * alpha)
                for level in levels:
                    locals()['_'+level] = (1 - alpha) *  locals()[level][i:i+1] + alpha * locals()[level][i+1:i+2]
                inputs = []
                for level in default_levels[1:]:
                    inputs.append(locals()['_'+level])
                img_pd = G(inputs, locals()['_'+default_levels[0]], True)
                img = np.uint8(img_pd.numpy().clip(0,1)*255)[0].transpose([1,2,0])
                imgs.append(Image.fromarray(img))
                stdout.write(f'{i*40+j+1}/{40*batch_size}\r')
                stdout.flush()
        print('')
        for i, img in enumerate(imgs):
            img.save(os.path.join(img_save_dir, str(i).zfill(5)+'.png'))
        imgs[0].save(out_path+'.gif', save_all=True, append_images=imgs[1:], duration=40, loop=0)
        out_path = out_path + '.mp4'
        os.system(f'ffmpeg -r 40 -i {img_save_dir}/%05d.png -hide_banner -loglevel warning -nostats -c:v libx264 -crf 23 -y {out_path}')
        os.system(f'rm -rf {img_save_dir}')
