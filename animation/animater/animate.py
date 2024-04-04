import math
import imageio
import os
import copy
import torch
import torch.nn.functional as F
import torch.utils.checkpoint
import torchvision.transforms as T
import numpy as np
from PIL import Image
from accelerate.utils import set_seed
from diffusers.models import AutoencoderKL
from diffusers import DPMSolverMultistepScheduler, DDPMScheduler
from diffusers.image_processor import VaeImageProcessor
from diffusers.utils.import_utils import is_xformers_available
from diffusers.models.attention_processor import AttnProcessor2_0
from diffusers.models.attention import BasicTransformerBlock
from transformers import CLIPTextModel, CLIPTokenizer
from einops import rearrange
from animation.animater.animate_anything.models import LatentToVideoPipeline, UNet3DConditionModel
from animation.animater.animate_anything.utils import calculate_motion_precision, calculate_latent_motion_score, \
    DDPM_forward_timesteps, tensor_to_vae_latent
from animation.logger import logger


class Function:
    def __init__(self):
        pass

    @staticmethod
    def cast_to_gpu_and_type(model_list, device, weight_dtype):
        for model in model_list:
            if model is not None: model.to(device, dtype=weight_dtype)

    @staticmethod
    def is_attn(name):
        return name.split('.')[-1] in ('attn1', 'attn2')

    @staticmethod
    def set_processors(attentions):
        for attn in attentions: attn.set_processor(AttnProcessor2_0())

    def set_torch_2_attn(self, unet):
        optim_count = 0

        for name, module in unet.named_modules():
            if self.is_attn(name) and isinstance(module, torch.nn.ModuleList):
                for m in module:
                    if isinstance(m, BasicTransformerBlock):
                        self.set_processors([m.attn1, m.attn2])
                        optim_count += 1
        if optim_count > 0:
            logger.info(f"{optim_count} Attention layers using Scaled Dot Product Attention.")

    def handle_memory_attention(self, enable_xformers_memory_efficient_attention, enable_torch_2_attn, unet):
        try:
            is_torch_2 = hasattr(F, 'scaled_dot_product_attention')
            enable_torch_2 = is_torch_2 and enable_torch_2_attn

            if enable_xformers_memory_efficient_attention and not enable_torch_2:
                if is_xformers_available():
                    from xformers.ops import MemoryEfficientAttentionFlashAttentionOp
                    unet.enable_xformers_memory_efficient_attention(
                        attention_op=MemoryEfficientAttentionFlashAttentionOp)
                else:
                    raise ValueError("xformers is not available. Make sure it is installed correctly")

            if enable_torch_2:
                self.set_torch_2_attn(unet)
        except:
            logger.info("Could not enable memory efficient attention for xformers or Torch 2.0.")


class PrimaryModels(Function):
    def __init__(self, pretrained_model_path, motion_strength):
        super().__init__()
        self.pretrained_model_path = pretrained_model_path
        self.motion_strength = motion_strength

    def load_primary_models(self, in_channels=-1):
        noise_scheduler = DDPMScheduler.from_pretrained(self.pretrained_model_path, subfolder="scheduler")
        tokenizer = CLIPTokenizer.from_pretrained(self.pretrained_model_path, subfolder="tokenizer")
        text_encoder = CLIPTextModel.from_pretrained(self.pretrained_model_path, subfolder="text_encoder")
        vae = AutoencoderKL.from_pretrained(self.pretrained_model_path, subfolder="vae")
        unet = UNet3DConditionModel.from_pretrained(self.pretrained_model_path, subfolder="unet")
        if 0 < in_channels != unet.config.in_channels:
            # first time init, modify unet conv in
            unet2 = unet
            unet = UNet3DConditionModel.from_pretrained(self.pretrained_model_path, subfolder="unet",
                                                        in_channels=in_channels,
                                                        low_cpu_mem_usage=False, device_map=None,
                                                        ignore_mismatched_sizes=True,
                                                        motion_strength=self.motion_strength)
            unet.conv_in.bias.data = copy.deepcopy(unet2.conv_in.bias)
            torch.nn.init.zeros_(unet.conv_in.weight)
            load_in_channel = unet2.conv_in.weight.data.shape[1]
            unet.conv_in.weight.data[:, in_channels - load_in_channel:] = copy.deepcopy(unet2.conv_in.weight.data)
            del unet2

        return noise_scheduler, tokenizer, text_encoder, vae, unet


class GenerativeMotion(PrimaryModels):
    def __init__(self, pretrained_model_path, prompt_image, prompt, mask, height=512, width=512, motion_strength=False):
        super().__init__(pretrained_model_path=pretrained_model_path, motion_strength=motion_strength)
        self.pretrained_model_path = pretrained_model_path
        self.prompt_image = prompt_image
        self.prompt = prompt
        self.mask = mask
        self.height = height
        self.width = width

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

    @staticmethod
    def freeze_models(models_to_freeze):
        for model in models_to_freeze:
            if model is not None: model.requires_grad_(False)

    @staticmethod
    def eval(pipeline, vae_processor, validation_data, out_file, index, forward_t=35, preview=True):
        vae = pipeline.vae
        diffusion_scheduler = pipeline.scheduler
        device = vae.device
        dtype = vae.dtype
        prompt = validation_data['prompt']
        pimg = Image.open(validation_data['prompt_image'])
        if pimg.mode == "RGBA":
            pimg = pimg.convert("RGB")
        width, height = pimg.size
        scale = math.sqrt(width * height / (validation_data['height'] * validation_data['width']))
        validation_data['height'] = round(height / scale / 8) * 8
        validation_data['width'] = round(width / scale / 8) * 8
        input_image = vae_processor.preprocess(pimg, validation_data['height'], validation_data['width'])
        input_image = input_image.unsqueeze(0).to(dtype).to(device)
        input_image_latents = tensor_to_vae_latent(input_image, vae)

        if 'mask' in validation_data:
            mask = Image.open(validation_data['mask'])
            mask = mask.resize((validation_data['width'], validation_data['height']))
            np_mask = np.array(mask)
            np_mask[np_mask != 0] = 255
            np_mask = np_mask.astype('uint8')
        else:
            np_mask = np.ones([validation_data['height'], validation_data['width']], dtype=np.uint8) * 255
        out_mask_path = os.path.splitext(out_file)[0] + "_mask.jpg"
        Image.fromarray(np_mask).save(out_mask_path)

        initial_latents, time_steps = DDPM_forward_timesteps(input_image_latents, forward_t,
                                                             validation_data['num_frames'],
                                                             diffusion_scheduler)
        masks = T.ToTensor()(np_mask).to(dtype).to(device)
        # b,c,f,h,w
        _, _, _, h, w = initial_latents.shape
        masks = T.Resize([h, w], antialias=False)(masks)
        masks = rearrange(masks, 'b h w -> b 1 1 h w')
        motion_strength = validation_data.get("strength", index + 3)
        with torch.no_grad():
            video_frames, video_latents = pipeline(
                prompt=prompt,
                latents=initial_latents,
                width=validation_data['width'],
                height=validation_data['height'],
                num_frames=validation_data['num_frames'],
                num_inference_steps=validation_data['num_inference_steps'],
                guidance_scale=validation_data['guidance_scale'],
                condition_latent=input_image_latents,
                mask=masks,
                motion=[motion_strength],
                return_dict=False,
                timesteps=time_steps,
            )
        if preview:
            fps = validation_data['fps']
            imageio.mimwrite(out_file, video_frames, duration=int(1000 / fps), loop=0)
            imageio.mimwrite(out_file.replace('gif', 'mp4'), video_frames, fps=fps)
        real_motion_strength = calculate_latent_motion_score(video_latents).cpu().numpy()[0]
        precision = calculate_motion_precision(video_frames, np_mask)
        logger.info(
            f"save file {out_file}, motion strength {motion_strength} -> {real_motion_strength}, motion precision {precision}")

        del pipeline
        torch.cuda.empty_cache() if torch.cuda.is_available() else logger.info('CUDA is not available. Using CPU')
        return precision

    def batch_eval(self, unet, text_encoder, vae, vae_processor, pretrained_model_path,
                   validation_data, output_dir, preview, global_step=0, iters=6):
        device = vae.device
        unet.eval()
        text_encoder.eval()
        pipeline = LatentToVideoPipeline.from_pretrained(
            pretrained_model_path,
            text_encoder=text_encoder,
            vae=vae,
            unet=unet
        )

        diffusion_scheduler = DPMSolverMultistepScheduler.from_config(pipeline.scheduler.config)
        diffusion_scheduler.set_timesteps(validation_data['num_inference_steps'], device=device)
        pipeline.scheduler = diffusion_scheduler
        motion_precision = 0
        best_out_file = ""
        for t in range(iters):
            name = os.path.basename(validation_data['prompt_image'])
            out_file_dir = f"{output_dir}/{name.split('.')[0]}"
            os.makedirs(out_file_dir, exist_ok=True)
            out_file = f"{out_file_dir}/{global_step + t}.gif"
            precision = self.eval(pipeline, vae_processor,
                                  validation_data, out_file, t, forward_t=validation_data['num_inference_steps'],
                                  preview=preview)
            motion_precision += precision
            if np.amax(precision):
                best_out_file = out_file.replace('gif', 'mp4')
        motion_precision = motion_precision / iters
        print(validation_data['prompt_image'], "precision", motion_precision)
        del pipeline
        return best_out_file

    def render(
            self,
            num_frames,
            num_inference_steps,
            guidance_scale,
            fps,
            strength,
            enable_xformers_memory_efficient_attention: bool = False,
            enable_torch_2_attn: bool = False,
            seed=None
    ):
        if seed is not None:
            set_seed(seed)

        validation_data = self.load_data(num_frames, num_inference_steps, guidance_scale, fps, strength)
        _, _, text_encoder, vae, unet = self.load_primary_models()
        vae_processor = VaeImageProcessor()
        # Freeze any necessary models
        self.freeze_models([vae, text_encoder, unet])

        # Enable xformers if available
        self.handle_memory_attention(enable_xformers_memory_efficient_attention, enable_torch_2_attn, unet)

        # Enable VAE slicing to save memory.
        vae.enable_slicing()

        # For mixed precision training we cast the text_encoder and vae weights to half-precision
        # as these models are only used for inference, keeping weights in full precision is not required.
        weight_dtype = torch.half if torch.cuda.is_available() else torch.float32  # half for GPU and float32 for CPU

        # Move text encoders, and VAE to GPU
        models_to_cast = [text_encoder, unet, vae]
        output_path = os.path.join(os.getcwd(), 'output')
        os.makedirs(output_path, exist_ok=True)
        self.cast_to_gpu_and_type(models_to_cast, torch.device("cuda") if torch.cuda.is_available() else 'cpu',
                                  weight_dtype)
        final_vid = self.batch_eval(unet, text_encoder, vae, vae_processor, self.pretrained_model_path,
                                    validation_data, output_path, True)
        return final_vid


if __name__ == "__main__":
    pass
    # parser = argparse.ArgumentParser()
    # parser.add_argument("--pretrained_path", type=str, default="put your pretrained path here")
    # parser.add_argument("--image", type=str, default=None)
    # parser.add_argument("--text", type=str, default=None)
    # parser.add_argument("--mask", type=str, default=None)
    # parser.add_argument("--eval", action="store_true")
    # parser.add_argument('rest', nargs=argparse.REMAINDER)
    # args = parser.parse_args()
    # if args.eval:
    #     GenerativeMotion(args.pretrained_path, args.image, args.text, args.mask).render()
