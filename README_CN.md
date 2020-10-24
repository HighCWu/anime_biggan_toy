# 动漫BigGAN玩具

使用BigGAN生成令人惊叹的动漫图片。玩的开心！！！

[EN](README.md) | 中文

![](imgs/std_out.gif)

## 免费的V100 GPU用于模型测试

[我的AI Studio项目](https://aistudio.baidu.com/aistudio/projectdetail/1141070)

你可以下载PaddlePaddle版本的预训练模型在本地玩耍 [Anime BigGAN Pretrained Models As A Dataset](https://aistudio.baidu.com/aistudio/datasetdetail/49029) 或者 [Google Drive](https://drive.google.com/file/d/1V9emQcBOz1ujrlGGDxYFsdavWbUxG1ws/view?usp=sharing)，我只在谷歌硬盘分享了生成器.

文件`Samples.ipynb`是我的AI Studio项目的notebook副本。

我的QQ交流群：1044867291

## 在谷歌的Colab中玩耍

[即刻开始](https://colab.research.google.com/github/HighCWu/anime_biggan_toy/blob/main/colab/Play_Anime_BigGAN.ipynb)！

## 把预训练模型转换到TFHub, PyTorch and PaddlePaddle的步骤

- 1. [把Shawwn的模型转换为TFHub版本](https://colab.research.google.com/github/HighCWu/anime_biggan_toy/blob/main/colab/AnimeBigGAN_tf2hub.ipynb).
- 2. 如果你想直接就从TFHub的模型开始玩，来[这儿](https://colab.research.google.com/github/HighCWu/anime_biggan_toy/blob/main/colab/Anime_BigGAN_Demo.ipynb).
- 3. [把TFHub的生成器模型转换到PyTorch](https://colab.research.google.com/github/HighCWu/anime_biggan_toy/blob/main/colab/pytorch_anime_biggan_for_generator_converter.ipynb).
- 4. [把TFHub的判别器模型转换到PyTorch](https://colab.research.google.com/github/HighCWu/anime_biggan_toy/blob/main/colab/pytorch_anime_biggan_for_discriminator_converter.ipynb).
- 5. 如果你想从PyTorch的模型开始玩，来[这儿](https://colab.research.google.com/github/HighCWu/anime_biggan_toy/blob/main/colab/pytorch_anime_biggan.ipynb).
- 6. [把TFHub的生成器模型转换到PaddlePaddle](https://colab.research.google.com/github/HighCWu/anime_biggan_toy/blob/main/colab/paddle_anime_biggan_for_generator_converter.ipynb).
- 7. [把TFHub的判别器模型转换到PaddlePaddle](https://colab.research.google.com/github/HighCWu/anime_biggan_toy/blob/main/colab/paddle_anime_biggan_for_discriminator_converter.ipynb).
- 8. 如果你想从PaddlePaddle的模型开始玩，来[这儿](https://colab.research.google.com/github/HighCWu/anime_biggan_toy/blob/main/colab/paddle_anime_biggan.ipynb).
- 希望你玩得愉快并从中找到点新的玩法。

## TODO

- [x] Test code release. 
- [x] Make my AI Studio project public.
- [x] Publish the colab notebook of model converting from TF to PaddlePaddle and PyTorch.

## 参考

Models converted from Shawwn's training [1] and Gwern's release [2]. 2D character picture (HatsuneMiku) is licensed under CC BY-NC by piapro [3].

- [1] Shawwn, The model is trained based on his fork of google's 'compare_gan', https://github.com/shawwn/compare_gan/
- [2] Gwern, "A prototype anime BigGAN 256px trained on Danbooru2019+e621 for 600k iterations is now available for download", https://www.gwern.net/Faces#danbooru2019e621-256px-biggan
- [3] "For Creators", http://piapro.net/en_for_creators.html
