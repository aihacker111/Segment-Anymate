import imageio
import math
import numpy as np
import os
import torch
import torch.utils.checkpoint
import torchvision.transforms as T
from PIL import Image
from diffusers.image_processor import VaeImageProcessor
from diffusers import StableVideoDiffusionPipeline
from einops import rearrange
from animation.animater.animate_anything.models import LatentToVideoPipeline, TextStableVideoDiffusionPipeline
from animation.animater.animate_anything.utils import tensor_to_vae_latent, DDPM_forward
from einops import repeat


class AnimateModels:
    def __init__(self, pretrained_model_path, model_type):
        self.pretrained_model_path = pretrained_model_path
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.model_type = model_type

    def load_animate_model(self):
        if self.model_type == 'animate_anything':
            pipeline = LatentToVideoPipeline.from_pretrained(self.pretrained_model_path,
                                                             torch_dtype=torch.float16,
                                                             variant="fp16").to(
                self.device) if self.device == 'cuda' \
                else LatentToVideoPipeline.from_pretrained(self.pretrained_model_path,
                                                           torch_dtype=torch.float32).to(self.device)
            return pipeline
        elif self.model_type == 'animate_svd':
            pipeline = StableVideoDiffusionPipeline.from_pretrained(self.pretrained_model_path,
                                                                    torch_dtype=torch.float16,
                                                                    variant="fp16").to(
                self.device) if self.device == 'cuda' \
                else StableVideoDiffusionPipeline.from_pretrained(self.pretrained_model_path,
                                                                  torch_dtype=torch.float32).to(self.device)
            return pipeline


class GenerativeMotion(AnimateModels):
    def __init__(self, model_type, pretrained_model_path, prompt_image, prompt, mask, seed, height=512, width=512,
                 motion_strength=False):
        super().__init__(pretrained_model_path=pretrained_model_path, model_type=model_type)
        self.pretrained_model_path = pretrained_model_path
        self.prompt_image = prompt_image
        self.prompt = prompt
        self.mask = mask
        self.height = height
        self.width = width
        self.seed = seed
        self.sample_idx = 0
        self.output_dir = os.getcwd()
        self.model_type = model_type

    def load_data(self, num_frames, num_inference_steps, guidance_scale, fps, strength, motion_bucket_id,
                  decode_chunk_size):
        if self.model_type == 'animate_anything':
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
        elif self.model_type == 'animate_svd':
            validation_data = {
                'prompt_image': self.prompt_image,
                'mask': self.mask,
                'height': self.height,
                'width': self.width,
                'num_frames': num_frames,
                'num_inference_steps': num_inference_steps,
                'guidance_scale': guidance_scale,
                'fps': fps,
                'motion_bucket_id': motion_bucket_id,
                'decode_chunk_size': decode_chunk_size
            }
            return validation_data

    def preprocessing(self, validation_data, vae, vae_processor, dtype, device):
        if self.model_type == 'animate_anything':
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
            np_mask = Image.open(validation_data['mask'])
            np_mask = np_mask.resize((validation_data['width'], validation_data['height']))
            np_mask = np.array(np_mask)
            np_mask[np_mask != 0] = 255
            np_mask = np_mask.astype('uint8')
            Image.fromarray(np_mask).save(self.mask)
            input_image_latents = tensor_to_vae_latent(input_image, vae)
            return pimg, block_size, input_image_latents, np_mask, height, width
        elif self.model_type == 'animate_svd':
            f = validation_data['num_frames']
            pimg = Image.open(validation_data['prompt_image'])
            if pimg.mode == "RGBA":
                pimg = pimg.convert("RGB")
            np_mask = Image.open(validation_data['mask'])
            np_mask = np.array(np_mask)
            np_mask[np_mask != 0] = 255
            if np_mask.sum() == 0:
                np_mask[:] = 255
            width, height = pimg.size
            scale = math.sqrt(width * height / (validation_data['height'] * validation_data['width']))
            block_size = 64
            height = round(height / scale / block_size) * block_size
            width = round(width / scale / block_size) * block_size
            input_image = vae_processor.preprocess(pimg, height, width)
            input_image = input_image.to(dtype).to(device)
            input_image_latents = vae.encode(input_image).latent_dist.mode() * vae.config.scaling_factor
            return pimg, f, input_image_latents, np_mask, height, width

    def render(
            self,
            num_frames,
            num_inference_steps,
            guidance_scale,
            fps,
            strength,
            motion_bucket_id,
            decode_chunk_size
    ):
        if self.seed != "-1" and self.seed != "":
            torch.manual_seed(int(self.seed))
        else:
            torch.seed()
        self.seed = torch.initial_seed()
        vae_processor = VaeImageProcessor()
        save_sample_path = os.path.join(
            self.output_dir, f"{self.sample_idx}.mp4")
        validation_data = self.load_data(num_frames, num_inference_steps, guidance_scale, fps, strength,
                                         motion_bucket_id,
                                         decode_chunk_size)
        if self.model_type == 'animate_anything':
            pipeline = self.load_animate_model()
            diffusion_scheduler = pipeline.scheduler
            vae = pipeline.vae
            device = vae.device
            dtype = vae.dtype
            _, _, input_image_latents, np_mask, height, width = self.preprocessing(validation_data, vae, vae_processor, dtype,
                                                                             device)
            b, c, _, h, w = input_image_latents.shape
            initial_latents, time_steps = DDPM_forward(input_image_latents,
                                                       num_inference_steps, validation_data['num_frames'],
                                                       diffusion_scheduler)
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
            imageio.mimwrite(save_sample_path.replace('mp4', 'gif'), video_frames, fps=validation_data['fps'])
            self.sample_idx += 1
            return save_sample_path

        elif self.model_type == 'animate_svd':
            pipeline = self.load_animate_model()
            vae = pipeline.vae
            device = vae.device
            dtype = vae.dtype
            pimg, f, input_image_latents, np_mask, height, width = self.preprocessing(validation_data, vae,
                                                                                      vae_processor, dtype, device)
            latents = repeat(input_image_latents, 'b c h w->b f c h w', f=f)

            b, f, c, h, w = latents.shape

            mask = T.ToTensor()(np_mask).to(dtype).to(device)
            mask = T.Resize([h, w], antialias=False)(mask)
            mask = repeat(mask, 'b h w -> b f 1 h w', f=f).detach().clone()
            mask[:, 0] = 0
            freeze = repeat(latents[:, 0], 'b c h w -> b f c h w', f=f)
            condition_latents = latents * (1 - mask) + freeze * mask
            condition_latents = condition_latents / vae.config.scaling_factor

            motion_mask = pipeline.unet.config.in_channels == 9
            decode_chunk_size = validation_data.get("decode_chunk_size", 7)
            fps = validation_data.get("fps", 7)
            motion_bucket_id = validation_data.get("motion_bucket_id", 127)
            if motion_mask:
                video_frames = TextStableVideoDiffusionPipeline.__call__(
                    pipeline,
                    image=pimg,
                    width=width,
                    height=height,
                    num_frames=validation_data['num_frames'],
                    num_inference_steps=validation_data['num_inference_steps'],
                    decode_chunk_size=decode_chunk_size,
                    fps=fps,
                    motion_bucket_id=motion_bucket_id,
                    mask=mask,
                    condition_type="image",
                    condition_latent=condition_latents
                ).frames[0]
            else:
                video_frames = pipeline(
                    image=pimg,
                    width=width,
                    height=height,
                    num_frames=validation_data['num_frames'],
                    num_inference_steps=validation_data['num_inference_steps'],
                    fps=validation_data['fps'],
                    decode_chunk_size=validation_data['decode_chunk_size'],
                    motion_bucket_id=validation_data['motion_bucket_id'],
                ).frames[0]

            save_sample_path = os.path.join(
                self.output_dir, f"{self.sample_idx}.mp4")
            Image.fromarray(np_mask).save(self.mask)
            imageio.mimwrite(save_sample_path, video_frames, fps=validation_data['fps'])
            imageio.mimwrite(save_sample_path.replace('mp4', 'gif'), video_frames, fps=validation_data['fps'])
            self.sample_idx += 1
            return save_sample_path


if __name__ == "__main__":
    pass
