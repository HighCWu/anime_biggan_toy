# Anime BigGAN Toy

Generate Amazing Anime Pictures With BigGAN. Just Have Fun !!!

![](imgs/std_out.gif)

## Free V100 GPU For Model Test

[My AI Studio Project](https://aistudio.baidu.com/aistudio/projectdetail/1141070)

You can download the pretrained model of PaddlePaddle for local play from [Anime BigGAN Pretrained Models As A Dataset](https://aistudio.baidu.com/aistudio/datasetdetail/49029) or [Google Drive](https://drive.google.com/file/d/1V9emQcBOz1ujrlGGDxYFsdavWbUxG1ws/view?usp=sharing) which just contain the Generator Parameters.

File `Samples.ipynb` is a backup of this project's notebook.

## TODO

- [x] Test code release. 
- [ ] Make my AI Studio project public.
- [ ] Publish the colab notebook of model converting from TF to PaddlePaddle and PyTorch.


## References

Models converted from Shawwn's training [1] and Gwern's release [2]. 2D character picture (HatsuneMiku) is licensed under CC BY-NC by piapro [3].

- [1] Shawwn, The model is trained based on his fork of google's 'compare_gan', https://github.com/shawwn/compare_gan/
- [2] Gwern, "A prototype anime BigGAN 256px trained on Danbooru2019+e621 for 600k iterations is now available for download", https://www.gwern.net/Faces#danbooru2019e621-256px-biggan
- [3] "For Creators", http://piapro.net/en_for_creators.html
