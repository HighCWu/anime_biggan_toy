# Anime BigGAN Toy

[![Baidu AI Studio](https://img.shields.io/static/v1?label=Baidu&message=AI%20Studio%20Free%20A100&color=blue)](https://aistudio.baidu.com/aistudio/projectdetail/1141070)

Generate Amazing Anime Pictures With BigGAN. Just Have Fun !!!

EN | [中文](README_CN.md)

![](imgs/std_out.gif)

## Free V100 GPU For Model Test

[My AI Studio Project](https://aistudio.baidu.com/aistudio/projectdetail/1141070)

You can download the pretrained model of PaddlePaddle for local play from [Anime BigGAN Pretrained Models As A Dataset](https://aistudio.baidu.com/aistudio/datasetdetail/49029) or [Google Drive](https://drive.google.com/file/d/1V9emQcBOz1ujrlGGDxYFsdavWbUxG1ws/view?usp=sharing) which just contain the Generator Parameters.

File `Samples.ipynb` is a backup of this project's notebook.

My QQ Group: 1044867291

My Discord: https://discord.gg/YwWcAS47qb

## Play in colab

[Play it right now](https://colab.research.google.com/github/HighCWu/anime_biggan_toy/blob/main/colab/Play_Anime_BigGAN.ipynb)!

[Play with TF](https://colab.research.google.com/github/HighCWu/anime_biggan_toy/blob/main/colab/AnimeBigGAN_tf2hub_fast.ipynb)!

## Steps to convert model to TFHub, PyTorch and PaddlePaddle

- 1. [Convert Shawwn's Model to TFHub version](https://colab.research.google.com/github/HighCWu/anime_biggan_toy/blob/main/colab/AnimeBigGAN_tf2hub_fast.ipynb).
- 2. If you want to play the TFHub model, come [here](https://colab.research.google.com/github/HighCWu/anime_biggan_toy/blob/main/colab/Anime_BigGAN_Demo.ipynb).
- 3. [Convert TFHub generator model to PyTorch](https://colab.research.google.com/github/HighCWu/anime_biggan_toy/blob/main/colab/pytorch_anime_biggan_for_generator_converter.ipynb).
- 4. [Convert TFHub discrimator model to PyTorch](https://colab.research.google.com/github/HighCWu/anime_biggan_toy/blob/main/colab/pytorch_anime_biggan_for_discriminator_converter.ipynb).
- 5. If you want to play the PyTorch model, come [here](https://colab.research.google.com/github/HighCWu/anime_biggan_toy/blob/main/colab/pytorch_anime_biggan.ipynb).
- 6. [Convert TFHub generator model to PaddlePaddle](https://colab.research.google.com/github/HighCWu/anime_biggan_toy/blob/main/colab/paddle_anime_biggan_for_generator_converter.ipynb).
- 7. [Convert TFHub discrimator model to PaddlePaddle](https://colab.research.google.com/github/HighCWu/anime_biggan_toy/blob/main/colab/paddle_anime_biggan_for_discriminator_converter.ipynb).
- 8. If you want to play the PaddlePaddle model, come [here](https://colab.research.google.com/github/HighCWu/anime_biggan_toy/blob/main/colab/paddle_anime_biggan.ipynb).
- Wish you can have fun and find out more interesting ways to play it.

**Update 2020-10-25: Add truncnorm. Add a fast way to play the source tf pretrained model.

**Update 2022-05-22: Add colab for directly converting tf model to pytorch.

## TODO

- [x] Test code release. 
- [x] Make my AI Studio project public.
- [x] Publish the colab notebook of model converting from TF to PaddlePaddle and PyTorch.
- [ ] Publish a gradio demo on HuggingFace Spaces.
- [ ] Update PaddlePaddle version to 2.3.

## References

Models converted from Shawwn's training [1] and Gwern's release [2]. 2D character picture (HatsuneMiku) is licensed under CC BY-NC by piapro [3].

- [1] Shawwn, The model is trained based on his fork of google's 'compare_gan', https://github.com/shawwn/compare_gan/
- [2] Gwern, "A prototype anime BigGAN 256px trained on Danbooru2019+e621 for 600k iterations is now available for download", https://www.gwern.net/Faces#danbooru2019e621-256px-biggan
- [3] "For Creators", http://piapro.net/en_for_creators.html
