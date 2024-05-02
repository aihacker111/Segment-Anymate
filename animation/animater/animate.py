import math
import imageio
import numpy as np
import os
import torch
import torch.utils.checkpoint
import torchvision.transforms as T
from PIL import Image
from diffusers.image_processor import VaeImageProcessor
from einops import rearrange
from animation.animater.animate_anything.models import LatentToVideoPipeline
from animation.animater.animate_anything.utils import tensor_to_vae_latent, DDPM_forward


class AnimateModels:
    def __init__(self, pretrained_model_path):
        self.pretrained_model_path = pretrained_model_path
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'

    def load_animate_model(self):
        pipeline = LatentToVideoPipeline.from_pretrained(self.pretrained_model_path,
                                                         torch_dtype=torch.float16,
                                                         variant="fp16").to(self.device) if self.device == 'cuda' \
            else LatentToVideoPipeline.from_pretrained(self.pretrained_model_path,
                                                       torch_dtype=torch.float32).to(self.device)
        return pipeline


class GenerativeMotion(AnimateModels):
    def __init__(self, pretrained_model_path, prompt_image, prompt, mask, seed, height=512, width=512,motion_strength=False):
        super().__init__(pretrained_model_path=pretrained_model_path)
        self.pretrained_model_path = pretrained_model_path
        self.prompt_image = prompt_image
        self.prompt = prompt
        self.mask = mask
        self.height = height
        self.width = width
        self.seed = seed
        self.sample_idx = 0
        self.output_dir = os.getcwd()

    def load_data(self, num_frames, num_inference_steps, guidance_scale, fps, strength):
        validation_data = {
            'prompt_image': self.prompt_image,
            'prompt': self.prompt,
            'mask': self.mask,
            'height': self.height,
            'width': self.width,
            'sample_preview': True,
            'num_frames': num_frames,
            'num_inference_steps': num_inference_steps,
            'guidance_scale': guidance_scale,
            'fps': fps,
            'strength': strength
        }
        return validation_data

    def render(
            self,
            num_frames,
            num_inference_steps,
            guidance_scale,
            fps,
            strength
    ):
        if self.seed != "-1" and self.seed != "":
            torch.manual_seed(int(self.seed))
        else:
            torch.seed()
        self.seed = torch.initial_seed()
        pipeline = self.load_animate_model()
        vae = pipeline.vae
        diffusion_scheduler = pipeline.scheduler
        validation_data = self.load_data(num_frames, num_inference_steps, guidance_scale, fps, strength)
        vae_processor = VaeImageProcessor()

        device = vae.device
        dtype = vae.dtype
        pimg = Image.open(validation_data['prompt_image'])
        if pimg.mode == "RGBA":
            pimg = pimg.convert("RGB")
        width, height = pimg.size
        scale = math.sqrt(width * height / (validation_data['height'] * validation_data['width']))
        block_size = 8
        height = round(height / scale / block_size) * block_size
        width = round(width / scale / block_size) * block_size
        input_image = vae_processor.preprocess(pimg, height, width)
        input_image = input_image.unsqueeze(0).to(dtype).to(device)
        input_image_latents = tensor_to_vae_latent(input_image, vae)
        np_mask = Image.open(validation_data['mask'])
        np_mask = np.array(np_mask)
        np_mask[np_mask != 0] = 255
        if np_mask.sum() == 0:
            np_mask[:] = 255
        save_sample_path = os.path.join(
            self.output_dir, f"{self.sample_idx}.mp4")
        Image.fromarray(np_mask).save(self.mask)

        b, c, _, h, w = input_image_latents.shape
        initial_latents, time_steps = DDPM_forward(input_image_latents,
                                                  num_inference_steps, validation_data['num_frames'], diffusion_scheduler)
        mask = T.ToTensor()(np_mask).to(dtype).to(device)
        b, c, f, h, w = initial_latents.shape
        mask = T.Resize([h, w], antialias=False)(mask)
        mask = rearrange(mask, 'b h w -> b 1 1 h w')
        motion_strength = strength * mask.mean().item()
        print(f"outfile {save_sample_path}, prompt {validation_data['prompt']}, motion_strength {motion_strength}")
        with torch.no_grad():
            video_frames, _ = pipeline(
                prompt=validation_data['prompt'],
                latents=initial_latents,
                width=width,
                height=height,
                num_frames=validation_data['num_frames'],
                num_inference_steps=num_inference_steps,
                guidance_scale=guidance_scale,
                condition_latent=input_image_latents,
                mask=mask,
                motion=[motion_strength],
                return_dict=False,
                timesteps=time_steps,
            )

        imageio.mimwrite(save_sample_path, video_frames, fps=validation_data['fps'])
        imageio.mimwrite(save_sample_path.replace('gif', 'mp4'), video_frames, fps=validation_data['fps'])
        self.sample_idx += 1
        return save_sample_path


if __name__ == "__main__":
    pass
