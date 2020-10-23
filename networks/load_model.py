import paddle.fluid as fluid
from paddle.fluid import dygraph as dg
from .model import Generator, Discriminator, model_cache


def load_G(path='data/anime-biggan-256px-run39-607250.generator'):
    place = fluid.CUDAPlace(fluid.dygraph.ParallelEnv().dev_id)
    fluid.enable_dygraph(place)

    generator = Generator(code_dim=140, n_class=1000, chn=96, blocks_with_attention="B5", resolution=256)
    generator.set_dict(dg.load_dygraph(path)[0])
    model_cache.G = generator


def load_D(path='data/anime-biggan-256px-run39-607250.discriminator'):
    place = fluid.CUDAPlace(fluid.dygraph.ParallelEnv().dev_id)
    fluid.enable_dygraph(place)

    discriminator = Discriminator(n_class=1000, chn=96, blocks_with_attention="B2", resolution=256)
    discriminator.set_dict(dg.load_dygraph(path)[0])
    model_cache.D = discriminator
