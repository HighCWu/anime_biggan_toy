#!/usr/bin/env python

from __future__ import annotations

import argparse
import os
import pickle
import sys
from typing import List, Tuple

import gradio as gr
import numpy as np
import torch
import torch.nn as nn
from model import Generator
from huggingface_hub import hf_hub_download

from moviepy.editor import *


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument('--device', type=str, default='cpu')
    parser.add_argument('--theme', type=str)
    parser.add_argument('--share', action='store_true')
    parser.add_argument('--port', type=int)
    parser.add_argument('--disable-queue',
                        dest='enable_queue',
                        action='store_false')
    return parser.parse_args()

cache_mp4_path = [f'/tmp/{str(i).zfill(2)}.mp4' for i in range(50)]
path_iter = iter(cache_mp4_path)

class App:
    '''
    Construct refer to https://huggingface.co/spaces/Gradio-Blocks/StyleGAN-Human
    '''
    def __init__(self, device: torch.device):
        self.device = device
        self.model = self.load_model()

    def load_model(self) -> nn.Module:
        path = hf_hub_download('HighCWu/anime-biggan-pytorch',
                               f'pytorch_model.bin')
        state_dict = torch.load(path, map_location='cpu')
        model = Generator(
            code_dim=140, n_class=1000, chn=96, 
            blocks_with_attention="B5", resolution=256
        )
        model.load_state_dict(state_dict)
        model.eval()
        model.to(self.device)
        with torch.inference_mode():
            z = torch.zeros((1, model.z_dim)).to(self.device)
            label = torch.zeros([1, model.c_dim], device=self.device)
            label[:,0] = 1
            model(z, label)
        return model

    def get_levels(self) -> List[str]:
        return [f'Level {i}' for i in range(self.model.n_level)]

    def generate_z_label(self, z_dim: int, c_dim: int, seed: int) -> Tuple[torch.Tensor, torch.Tensor]:
        rng = np.random.RandomState(seed)
        z = rng.randn(
            1, z_dim)
        label = rng.randint(0, c_dim, size=(1,))
        z = torch.from_numpy(z).to(self.device).float()
        label = torch.from_numpy(label).to(self.device).long()
        label = torch.nn.functional.one_hot(label, 1000).float()
        return z, label

    @torch.inference_mode()
    def generate_single_image(self, seed: int) -> np.ndarray:
        seed = int(np.clip(seed, 0, np.iinfo(np.uint32).max))

        z, label = self.generate_z_label(self.model.z_dim, self.model.c_dim, seed)

        out = self.model(z, label)
        out = (out.permute(0, 2, 3, 1) * 255).clamp(0, 255).to(
            torch.uint8)
        return out[0].cpu().numpy()

    @torch.inference_mode()
    def generate_interpolated_images(
            self, seed0: int, seed1: int,
            num_intermediate: int, levels: List[str]) -> List[np.ndarray]:
        seed0 = int(np.clip(seed0, 0, np.iinfo(np.uint32).max))
        seed1 = int(np.clip(seed1, 0, np.iinfo(np.uint32).max))
        levels = [int(level.split(' ')[1]) for level in levels]

        z0, label0 = self.generate_z_label(self.model.z_dim, self.model.c_dim, seed0)
        z1, label1 = self.generate_z_label(self.model.z_dim, self.model.c_dim, seed1)
        vec = z1 - z0
        dvec = vec / (num_intermediate + 1)
        zs = [z0 + dvec * i for i in range(num_intermediate + 2)]

        vec = label1 - label0
        dvec = vec / (num_intermediate + 1)
        labels = [label0 + dvec * i for i in range(num_intermediate + 2)]

        res = []
        for z, label in zip(zs, labels):
            z0_split = list(torch.chunk(z0, self.model.n_level, 1))
            z_split = list(torch.chunk(z, self.model.n_level, 1))
            for j in levels:
                z_split[j] = z0_split[j]
            z = torch.cat(z_split, 1)
            out = self.model(z, label)
            out = (out.permute(0, 2, 3, 1) * 255).clamp(0, 255).to(
                torch.uint8)
            out = out[0].cpu().numpy()
            res.append(out)

        fps = 1 / (5 / len(res))
        video = ImageSequenceClip(res, fps=fps)
        global path_iter
        try:
            video_path = next(path_iter)
        except:
            path_iter = iter(cache_mp4_path)
            video_path = next(path_iter)
        video.write_videofile(video_path, fps=fps)
        
        return res, video_path


def main():
    args = parse_args()
    app = App(device=torch.device(args.device))

    with gr.Blocks(theme=args.theme) as demo:
        gr.Markdown('''<center><h1>Anime-BigGAN</h1></center>
This is a Gradio Blocks app of <a href="https://github.com/HighCWu/anime_biggan_toy">HighCWu/anime_biggan_toy in github</a>.
''')

        with gr.Row():
            with gr.Box():
                with gr.Column():
                    with gr.Row():
                        with gr.Column():
                            with gr.Row():
                                seed1 = gr.Number(value=128, label='Seed 1')
                            with gr.Row():
                                generate_button1 = gr.Button('Generate')
                            with gr.Row():
                                generated_image1 = gr.Image(type='numpy', shape=(256,256),
                                                            label='Generated Image 1')
                        with gr.Column():
                            with gr.Row():
                                seed2 = gr.Number(value=6886, label='Seed 2')
                            with gr.Row():
                                generate_button2 = gr.Button('Generate')
                            with gr.Row():
                                generated_image2 = gr.Image(type='numpy', shape=(256,256),
                                                            label='Generated Image 2')
                    
                    with gr.Row():
                        gr.Image(value='imgs/out1.png', type='filepath',
                                 interactive=False, label='Sample results 1')
                    with gr.Row():
                        gr.Image(value='imgs/out2.png', type='filepath',
                                 interactive=False, label='Sample results 2')
            
            with gr.Box():
                with gr.Column():
                    with gr.Row():
                        num_frames = gr.Slider(
                            0,
                            41,
                            value=7,
                            step=1,
                            label='Number of Intermediate Frames between image 1 and image 2')
                    with gr.Row():
                        level_choices = gr.CheckboxGroup(
                            choices=app.get_levels(),
                            label='Levels of latents to fix based on the first latent')
                    with gr.Row():
                        interpolate_button = gr.Button('Interpolate')

                    with gr.Row():
                        interpolated_images = gr.Gallery(label='Output Images')
                    with gr.Row():
                        interpolated_video = gr.Video(label='Output Video')

        gr.Markdown(
            '<center><img src="https://visitor-badge.glitch.me/badge?page_id=gradio-blocks.anime-biggan" alt="visitor badge"/></center>'
        )

        generate_button1.click(app.generate_single_image,
                               inputs=[seed1],
                               outputs=generated_image1)
        generate_button2.click(app.generate_single_image,
                               inputs=[seed2],
                               outputs=generated_image2)
        interpolate_button.click(app.generate_interpolated_images,
                                 inputs=[seed1, seed2, num_frames, level_choices],
                                 outputs=[interpolated_images, interpolated_video])

    demo.launch(
        enable_queue=args.enable_queue,
        server_port=args.port,
        share=args.share,
    )


if __name__ == '__main__':
    main()
