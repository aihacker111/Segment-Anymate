import os.path
from typing import Callable, Dict, List, Optional, Union
from einops import repeat
from diffusers import TextToVideoSDPipeline, StableVideoDiffusionPipeline
from diffusers.pipelines.text_to_video_synthesis.pipeline_text_to_video_synth import tensor2vid, \
    TextToVideoSDPipelineOutput
from diffusers.pipelines.stable_video_diffusion.pipeline_stable_video_diffusion import tensor2vid as svd_tensor2vid, \
    StableVideoDiffusionPipelineOutput
from diffusers.utils.torch_utils import randn_tensor
import PIL
import torch
import sys

if 'COLAB_JUPYTER_TRANSPORT' not in os.environ:
    is_colab = False
    sys.path.insert(0, './animation/animater/animate_anything')
    print("Running on Locally")
else:
    is_colab = True
    sys.path.insert(0, './Segment-Anymate/animation/animater/animate_anything')
    print("Running on Google Colab")


class LatentToVideoPipeline(TextToVideoSDPipeline):
    @torch.no_grad()
    def __call__(
            self,
            prompt=None,
            height=None,
            width=None,
            num_frames: int = 16,
            num_inference_steps: int = 50,
            guidance_scale=9.0,
            negative_prompt=None,
            eta: float = 0.0,
            generator=None,
            latents=None,
            prompt_embeds=None,
            negative_prompt_embeds=None,
            output_type="np",
            return_dict: bool = True,
            callback=None,
            callback_steps: int = 1,
            cross_attention_kwargs=None,
            condition_latent=None,
            mask=None,
            timesteps=None,
            motion=None,
    ):
        r"""
        Function invoked when calling the pipeline for generation.

        Args:
            prompt (`str` or `List[str]`, *optional*):
                The prompt or prompts to guide the video generation. If not defined, one has to pass `prompt_embeds`.
                instead.
            height (`int`, *optional*, defaults to self.unet.config.sample_size * self.vae_scale_factor):
                The height in pixels of the generated video.
            width (`int`, *optional*, defaults to self.unet.config.sample_size * self.vae_scale_factor):
                The width in pixels of the generated video.
            num_frames (`int`, *optional*, defaults to 16):
                The number of video frames that are generated. Defaults to 16 frames which at 8 frames per seconds
                amounts to 2 seconds of video.
            num_inference_steps (`int`, *optional*, defaults to 50):
                The number of denoising steps. More denoising steps usually lead to a higher quality videos at the
                expense of slower inference.
            guidance_scale (`float`, *optional*, defaults to 7.5):
                Guidance scale as defined in [Classifier-Free Diffusion Guidance](https://arxiv.org/abs/2207.12598).
                `guidance_scale` is defined as `w` of equation 2. of [Imagen
                Paper](https://arxiv.org/pdf/2205.11487.pdf). Guidance scale is enabled by setting `guidance_scale >
                1`. Higher guidance scale encourages to generate videos that are closely linked to the text `prompt`,
                usually at the expense of lower video quality.
            negative_prompt (`str` or `List[str]`, *optional*):
                The prompt or prompts not to guide the video generation. If not defined, one has to pass
                `negative_prompt_embeds` instead. Ignored when not using guidance (i.e., ignored if `guidance_scale` is
                less than `1`).
            eta (`float`, *optional*, defaults to 0.0):
                Corresponds to parameter eta (η) in the DDIM paper: https://arxiv.org/abs/2010.02502. Only applies to
                [`schedulers.DDIMScheduler`], will be ignored for others.
            generator (`torch.Generator` or `List[torch.Generator]`, *optional*):
                One or a list of [torch generator(s)](https://pytorch.org/docs/stable/generated/torch.Generator.html)
                to make generation deterministic.
            latents (`torch.FloatTensor`, *optional*):
                Pre-generated noisy latents, sampled from a Gaussian distribution, to be used as inputs for video
                generation. Can be used to tweak the same generation with different prompts. If not provided, a latents
                tensor will ge generated by sampling using the supplied random `generator`. Latents should be of shape
                `(batch_size, num_channel, num_frames, height, width)`.
            prompt_embeds (`torch.FloatTensor`, *optional*):
                Pre-generated text embeddings. Can be used to easily tweak text inputs, *e.g.* prompt weighting. If not
                provided, text embeddings will be generated from `prompt` input argument.
            negative_prompt_embeds (`torch.FloatTensor`, *optional*):
                Pre-generated negative text embeddings. Can be used to easily tweak text inputs, *e.g.* prompt
                weighting. If not provided, negative_prompt_embeds will be generated from `negative_prompt` input
                argument.
            output_type (`str`, *optional*, defaults to `"np"`):
                The output format of the generate video. Choose between `torch.FloatTensor` or `np.array`.
            return_dict (`bool`, *optional*, defaults to `True`):
                Whether or not to return a [`~pipelines.stable_diffusion.TextToVideoSDPipelineOutput`] instead of a
                plain tuple.
            callback (`Callable`, *optional*):
                A function that will be called every `callback_steps` steps during inference. The function will be
                called with the following arguments: `callback(step: int, timestep: int, latents: torch.FloatTensor)`.
            callback_steps (`int`, *optional*, defaults to 1):
                The frequency at which the `callback` function will be called. If not specified, the callback will be
                called at every step.
            cross_attention_kwargs (`dict`, *optional*):
                A kwargs dictionary that if specified is passed along to the `AttentionProcessor` as defined under
                `self.processor` in
                [diffusers.cross_attention](https://github.com/huggingface/diffusers/blob/main/src/diffusers/models/cross_attention.py).

        Examples:

        Returns:
            [`~pipelines.stable_diffusion.TextToVideoSDPipelineOutput`] or `tuple`:
            [`~pipelines.stable_diffusion.TextToVideoSDPipelineOutput`] if `return_dict` is True, otherwise a `tuple.
            When returning a tuple, the first element is a list with the generated frames.
        """
        # 0. Default height and width to unet
        height = height or self.unet.config.sample_size * self.vae_scale_factor
        width = width or self.unet.config.sample_size * self.vae_scale_factor

        num_images_per_prompt = 1
        # 1. Check inputs. Raise error if not correct
        self.check_inputs(
            prompt, height, width, callback_steps, negative_prompt, prompt_embeds, negative_prompt_embeds
        )

        # 2. Define call parameters
        if prompt is not None and isinstance(prompt, str):
            batch_size = 1
        elif prompt is not None and isinstance(prompt, list):
            batch_size = len(prompt)
        else:
            batch_size = prompt_embeds.shape[0]

        device = latents.device
        # here `guidance_scale` is defined analog to the guidance weight `w` of equation (2)
        # of the Imagen paper: https://arxiv.org/pdf/2205.11487.pdf . `guidance_scale = 1`
        # corresponds to doing no classifier free guidance.
        do_classifier_free_guidance = guidance_scale > 1.0

        # 3. Encode input prompt
        text_encoder_lora_scale = (
            cross_attention_kwargs.get("scale", None) if cross_attention_kwargs is not None else None
        )
        prompt_embeds = self._encode_prompt(
            prompt,
            device,
            num_images_per_prompt,
            do_classifier_free_guidance,
            negative_prompt,
            prompt_embeds=prompt_embeds,
            negative_prompt_embeds=negative_prompt_embeds,
            lora_scale=text_encoder_lora_scale,
        )

        # 4. Prepare timesteps
        self.scheduler.set_timesteps(num_inference_steps, device=device)
        if timesteps is None:
            timesteps = self.scheduler.timesteps
        else:
            num_inference_steps = len(timesteps)
        # 5. Prepare latent variables. do nothing

        # 6. Prepare extra step kwargs. TODO: Logic should ideally just be moved out of the pipeline
        extra_step_kwargs = self.prepare_extra_step_kwargs(generator, eta)

        # 7. Denoising loop
        num_warmup_steps = len(timesteps) - num_inference_steps * self.scheduler.order
        uncondition_latent = condition_latent
        condition_latent = torch.cat(
            [uncondition_latent, condition_latent]) if do_classifier_free_guidance else condition_latent
        with self.progress_bar(total=num_inference_steps) as progress_bar:
            for i, t in enumerate(timesteps):
                # expand the latents if we are doing classifier free guidance
                latent_model_input = torch.cat([latents] * 2) if do_classifier_free_guidance else latents
                latent_model_input = self.scheduler.scale_model_input(latent_model_input, t)
                if motion is not None:
                    motion = torch.tensor(motion, device=device)
                noise_pred = self.unet(
                    latent_model_input,
                    t,
                    encoder_hidden_states=prompt_embeds,
                    cross_attention_kwargs=cross_attention_kwargs,
                    condition_latent=condition_latent,
                    mask=mask,
                    motion=motion
                ).sample
                # perform guidance
                if do_classifier_free_guidance:
                    noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)
                    noise_pred = noise_pred_uncond + guidance_scale * (noise_pred_text - noise_pred_uncond)

                # reshape latents
                bsz, channel, frames, width, height = latents.shape
                latents = latents.permute(0, 2, 1, 3, 4).reshape(bsz * frames, channel, width, height)
                noise_pred = noise_pred.permute(0, 2, 1, 3, 4).reshape(bsz * frames, channel, width, height)

                # compute the previous noisy sample x_t -> x_t-1
                latents = self.scheduler.step(noise_pred, t, latents, **extra_step_kwargs).prev_sample

                # reshape latents back
                latents = latents[None, :].reshape(bsz, frames, channel, width, height).permute(0, 2, 1, 3, 4)

                # call the callback, if provided
                if i == len(timesteps) - 1 or ((i + 1) > num_warmup_steps and (i + 1) % self.scheduler.order == 0):
                    progress_bar.update()
                    if callback is not None and i % callback_steps == 0:
                        callback(i, t, latents)

        video_tensor = self.decode_latents(latents)

        if output_type == "pt":
            video = video_tensor
        else:
            video = tensor2vid(video_tensor)

        # Offload last model to CPU
        if hasattr(self, "final_offload_hook") and self.final_offload_hook is not None:
            self.final_offload_hook.offload()

        if not return_dict:
            return video, latents

        return TextToVideoSDPipelineOutput(frames=video)


def _append_dims(x, target_dims):
    """Appends dimensions to the end of a tensor until it has target_dims dimensions."""
    dims_to_append = target_dims - x.ndim
    if dims_to_append < 0:
        raise ValueError(f"input has {x.ndim} dims but target_dims is {target_dims}, which is less")
    return x[(...,) + (None,) * dims_to_append]


class MaskStableVideoDiffusionPipeline(StableVideoDiffusionPipeline):
    @torch.no_grad()
    def __call__(
            self,
            image,
            height: int = 576,
            width: int = 1024,
            num_frames: Optional[int] = None,
            num_inference_steps: int = 25,
            min_guidance_scale: float = 1.0,
            max_guidance_scale: float = 3.0,
            fps: int = 7,
            motion_bucket_id: int = 127,
            noise_aug_strength: int = 0.02,
            decode_chunk_size: Optional[int] = None,
            num_videos_per_prompt: Optional[int] = 1,
            generator: Optional[Union[torch.Generator, List[torch.Generator]]] = None,
            latents: Optional[torch.FloatTensor] = None,
            output_type: Optional[str] = "pil",
            callback_on_step_end: Optional[Callable[[int, int, Dict], None]] = None,
            callback_on_step_end_tensor_inputs: List[str] = ["latents"],
            return_dict: bool = True,
            mask=None,
    ):
        r"""
        The call function to the pipeline for generation.

        Args:
            image (`PIL.Image.Image` or `List[PIL.Image.Image]` or `torch.FloatTensor`):
                Image or images to guide image generation. If you provide a tensor, it needs to be compatible with
                [`CLIPImageProcessor`](https://huggingface.co/lambdalabs/sd-image-variations-diffusers/blob/main/feature_extractor/preprocessor_config.json).
            height (`int`, *optional*, defaults to `self.unet.config.sample_size * self.vae_scale_factor`):
                The height in pixels of the generated image.
            width (`int`, *optional*, defaults to `self.unet.config.sample_size * self.vae_scale_factor`):
                The width in pixels of the generated image.
            num_frames (`int`, *optional*):
                The number of video frames to generate. Defaults to 14 for `stable-video-diffusion-img2vid` and to 25 for `stable-video-diffusion-img2vid-xt`
            num_inference_steps (`int`, *optional*, defaults to 25):
                The number of denoising steps. More denoising steps usually lead to a higher quality image at the
                expense of slower inference. This parameter is modulated by `strength`.
            min_guidance_scale (`float`, *optional*, defaults to 1.0):
                The minimum guidance scale. Used for the classifier free guidance with first frame.
            max_guidance_scale (`float`, *optional*, defaults to 3.0):
                The maximum guidance scale. Used for the classifier free guidance with last frame.
            fps (`int`, *optional*, defaults to 7):
                Frames per second. The rate at which the generated images shall be exported to a video after generation.
                Note that Stable Diffusion Video's UNet was micro-conditioned on fps-1 during training.
            motion_bucket_id (`int`, *optional*, defaults to 127):
                The motion bucket ID. Used as conditioning for the generation. The higher the number the more motion will be in the video.
            noise_aug_strength (`int`, *optional*, defaults to 0.02):
                The amount of noise added to the init image, the higher it is the less the video will look like the init image. Increase it for more motion.
            decode_chunk_size (`int`, *optional*):
                The number of frames to decode at a time. The higher the chunk size, the higher the temporal consistency
                between frames, but also the higher the memory consumption. By default, the decoder will decode all frames at once
                for maximal quality. Reduce `decode_chunk_size` to reduce memory usage.
            num_videos_per_prompt (`int`, *optional*, defaults to 1):
                The number of images to generate per prompt.
            generator (`torch.Generator` or `List[torch.Generator]`, *optional*):
                A [`torch.Generator`](https://pytorch.org/docs/stable/generated/torch.Generator.html) to make
                generation deterministic.
            latents (`torch.FloatTensor`, *optional*):
                Pre-generated noisy latents sampled from a Gaussian distribution, to be used as inputs for image
                generation. Can be used to tweak the same generation with different prompts. If not provided, a latents
                tensor is generated by sampling using the supplied random `generator`.
            output_type (`str`, *optional*, defaults to `"pil"`):
                The output format of the generated image. Choose between `PIL.Image` or `np.array`.
            callback_on_step_end (`Callable`, *optional*):
                A function that calls at the end of each denoising steps during the inference. The function is called
                with the following arguments: `callback_on_step_end(self: DiffusionPipeline, step: int, timestep: int,
                callback_kwargs: Dict)`. `callback_kwargs` will include a list of all tensors as specified by
                `callback_on_step_end_tensor_inputs`.
            callback_on_step_end_tensor_inputs (`List`, *optional*):
                The list of tensor inputs for the `callback_on_step_end` function. The tensors specified in the list
                will be passed as `callback_kwargs` argument. You will only be able to include variables listed in the
                `._callback_tensor_inputs` attribute of your pipeline class.
            return_dict (`bool`, *optional*, defaults to `True`):
                Whether or not to return a [`~pipelines.stable_diffusion.StableDiffusionPipelineOutput`] instead of a
                plain tuple.

        Returns:
            [`~pipelines.stable_diffusion.StableVideoDiffusionPipelineOutput`] or `tuple`:
                If `return_dict` is `True`, [`~pipelines.stable_diffusion.StableVideoDiffusionPipelineOutput`] is returned,
                otherwise a `tuple` is returned where the first element is a list of list with the generated frames.

        Examples:

        ```py
        from diffusers import StableVideoDiffusionPipeline
        from diffusers.utils import load_image, export_to_video

        pipe = StableVideoDiffusionPipeline.from_pretrained("stabilityai/stable-video-diffusion-img2vid-xt", torch_dtype=torch.float16, variant="fp16")
        pipe.to("cuda")

        image = load_image("https://lh3.googleusercontent.com/y-iFOHfLTwkuQSUegpwDdgKmOjRSTvPxat63dQLB25xkTs4lhIbRUFeNBWZzYf370g=s1200")
        image = image.resize((1024, 576))

        frames = pipe(image, num_frames=25, decode_chunk_size=8).frames[0]
        export_to_video(frames, "generated.mp4", fps=7)
        ```
        """
        # 0. Default height and width to unet
        height = height or self.unet.config.sample_size * self.vae_scale_factor
        width = width or self.unet.config.sample_size * self.vae_scale_factor

        num_frames = num_frames if num_frames is not None else self.unet.config.num_frames
        decode_chunk_size = decode_chunk_size if decode_chunk_size is not None else num_frames

        # 1. Check inputs. Raise error if not correct
        self.check_inputs(image, height, width)

        # 2. Define call parameters
        if isinstance(image, PIL.Image.Image):
            batch_size = 1
        elif isinstance(image, list):
            batch_size = len(image)
        else:
            batch_size = image.shape[0]
        device = self._execution_device
        # here `guidance_scale` is defined analog to the guidance weight `w` of equation (2)
        # of the Imagen paper: https://arxiv.org/pdf/2205.11487.pdf . `guidance_scale = 1`
        # corresponds to doing no classifier free guidance.
        do_classifier_free_guidance = max_guidance_scale > 1.0

        # 3. Encode input image
        image_embeddings = self._encode_image(image, device, num_videos_per_prompt, do_classifier_free_guidance)

        # NOTE: Stable Diffusion Video was conditioned on fps - 1, which
        # is why it is reduced here.
        # See: https://github.com/Stability-AI/generative-models/blob/ed0997173f98eaf8f4edf7ba5fe8f15c6b877fd3/scripts/sampling/simple_video_sample.py#L188
        fps = fps - 1

        # 4. Encode input image using VAE
        image = self.image_processor.preprocess(image, height=height, width=width)
        noise = randn_tensor(image.shape, generator=generator, device=image.device, dtype=image.dtype)
        image = image + noise_aug_strength * noise

        needs_upcasting = self.vae.dtype == torch.float16 and self.vae.config.force_upcast
        if needs_upcasting:
            self.vae.to(dtype=torch.float32)

        image_latents = self._encode_vae_image(image, device, num_videos_per_prompt, do_classifier_free_guidance)
        image_latents = image_latents.to(image_embeddings.dtype)

        # cast back to fp16 if needed
        if needs_upcasting:
            self.vae.to(dtype=torch.float16)

        # Repeat the image latents for each frame so we can concatenate them with the noise
        # image_latents [batch, channels, height, width] ->[batch, num_frames, channels, height, width]
        image_latents = image_latents.unsqueeze(1).repeat(1, num_frames, 1, 1, 1)
        mask = repeat(mask, '1 h w -> 2 f 1 h w', f=num_frames)
        # 5. Get Added Time IDs
        added_time_ids = self._get_add_time_ids(
            fps,
            motion_bucket_id,
            noise_aug_strength,
            image_embeddings.dtype,
            batch_size,
            num_videos_per_prompt,
            do_classifier_free_guidance,
        )
        added_time_ids = added_time_ids.to(device)

        # 4. Prepare time_steps
        self.scheduler.set_timesteps(num_inference_steps, device=device)
        time_steps = self.scheduler.timesteps

        # 5. Prepare latent variables
        num_channels_latents = self.unet.config.in_channels
        latents = self.prepare_latents(
            batch_size * num_videos_per_prompt,
            num_frames,
            num_channels_latents,
            height,
            width,
            image_embeddings.dtype,
            device,
            generator,
            latents,
        )

        # 7. Prepare guidance scale
        guidance_scale = torch.linspace(min_guidance_scale, max_guidance_scale, num_frames).unsqueeze(0)
        guidance_scale = guidance_scale.to(device, latents.dtype)
        guidance_scale = guidance_scale.repeat(batch_size * num_videos_per_prompt, 1)
        guidance_scale = _append_dims(guidance_scale, latents.ndim)

        self._guidance_scale = guidance_scale

        # 8. Denoising loop
        num_warmup_steps = len(time_steps) - num_inference_steps * self.scheduler.order
        self._num_timesteps = len(time_steps)
        with self.progress_bar(total=num_inference_steps) as progress_bar:
            for i, t in enumerate(time_steps):
                # expand the latents if we are doing classifier free guidance
                latent_model_input = torch.cat([latents] * 2) if do_classifier_free_guidance else latents
                latent_model_input = self.scheduler.scale_model_input(latent_model_input, t)

                # Concatenate image_latents over channels dimention
                latent_model_input = torch.cat([mask, latent_model_input, image_latents], dim=2)

                # predict the noise residual
                noise_pred = self.unet(
                    latent_model_input,
                    t,
                    encoder_hidden_states=image_embeddings,
                    added_time_ids=added_time_ids,
                    return_dict=False,
                )[0]

                # perform guidance
                if do_classifier_free_guidance:
                    noise_pred_uncond, noise_pred_cond = noise_pred.chunk(2)
                    noise_pred = noise_pred_uncond + self.guidance_scale * (noise_pred_cond - noise_pred_uncond)

                # compute the previous noisy sample x_t -> x_t-1
                latents = self.scheduler.step(noise_pred, t, latents).prev_sample

                if callback_on_step_end is not None:
                    callback_kwargs = {}
                    for k in callback_on_step_end_tensor_inputs:
                        callback_kwargs[k] = locals()[k]
                    callback_outputs = callback_on_step_end(self, i, t, callback_kwargs)

                    latents = callback_outputs.pop("latents", latents)

                if i == len(time_steps) - 1 or ((i + 1) > num_warmup_steps and (i + 1) % self.scheduler.order == 0):
                    progress_bar.update()

        if output_type != "latent":
            # cast back to fp16 if needed
            if needs_upcasting:
                self.vae.to(dtype=torch.float16)
            frames = self.decode_latents(latents, num_frames, decode_chunk_size)
            frames = svd_tensor2vid(frames, self.image_processor, output_type=output_type)
        else:
            frames = latents

        self.maybe_free_model_hooks()

        if not return_dict:
            return frames

        return StableVideoDiffusionPipelineOutput(frames=frames)


class TextStableVideoDiffusionPipeline(StableVideoDiffusionPipeline):
    @torch.no_grad()
    def __call__(
        self,
        image,
        prompt_embeds = None,
        negative_prompt_embeds = None,
        height: int = 576,
        width: int = 1024,
        num_frames: Optional[int] = None,
        num_inference_steps: int = 25,
        min_guidance_scale: float = 1.0,
        max_guidance_scale: float = 3.0,
        fps: int = 7,
        motion_bucket_id: int = 127,
        noise_aug_strength: int = 0.02,
        decode_chunk_size: Optional[int] = None,
        num_videos_per_prompt: Optional[int] = 1,
        generator: Optional[Union[torch.Generator, List[torch.Generator]]] = None,
        latents: Optional[torch.FloatTensor] = None,
        output_type: Optional[str] = "pil",
        callback_on_step_end: Optional[Callable[[int, int, Dict], None]] = None,
        callback_on_step_end_tensor_inputs: List[str] = ["latents"],
        return_dict: bool = True,
        mask = None,
        condition_type = "image",
        condition_latent = None,
    ):
        r"""
        The call function to the pipeline for generation.

        Args:
            image (`PIL.Image.Image` or `List[PIL.Image.Image]` or `torch.FloatTensor`):
                Image or images to guide image generation. If you provide a tensor, it needs to be compatible with
                [`CLIPImageProcessor`](https://huggingface.co/lambdalabs/sd-image-variations-diffusers/blob/main/feature_extractor/preprocessor_config.json).
            height (`int`, *optional*, defaults to `self.unet.config.sample_size * self.vae_scale_factor`):
                The height in pixels of the generated image.
            width (`int`, *optional*, defaults to `self.unet.config.sample_size * self.vae_scale_factor`):
                The width in pixels of the generated image.
            num_frames (`int`, *optional*):
                The number of video frames to generate. Defaults to 14 for `stable-video-diffusion-img2vid` and to 25 for `stable-video-diffusion-img2vid-xt`
            num_inference_steps (`int`, *optional*, defaults to 25):
                The number of denoising steps. More denoising steps usually lead to a higher quality image at the
                expense of slower inference. This parameter is modulated by `strength`.
            min_guidance_scale (`float`, *optional*, defaults to 1.0):
                The minimum guidance scale. Used for the classifier free guidance with first frame.
            max_guidance_scale (`float`, *optional*, defaults to 3.0):
                The maximum guidance scale. Used for the classifier free guidance with last frame.
            fps (`int`, *optional*, defaults to 7):
                Frames per second. The rate at which the generated images shall be exported to a video after generation.
                Note that Stable Diffusion Video's UNet was micro-conditioned on fps-1 during training.
            motion_bucket_id (`int`, *optional*, defaults to 127):
                The motion bucket ID. Used as conditioning for the generation. The higher the number the more motion will be in the video.
            noise_aug_strength (`int`, *optional*, defaults to 0.02):
                The amount of noise added to the init image, the higher it is the less the video will look like the init image. Increase it for more motion.
            decode_chunk_size (`int`, *optional*):
                The number of frames to decode at a time. The higher the chunk size, the higher the temporal consistency
                between frames, but also the higher the memory consumption. By default, the decoder will decode all frames at once
                for maximal quality. Reduce `decode_chunk_size` to reduce memory usage.
            num_videos_per_prompt (`int`, *optional*, defaults to 1):
                The number of images to generate per prompt.
            generator (`torch.Generator` or `List[torch.Generator]`, *optional*):
                A [`torch.Generator`](https://pytorch.org/docs/stable/generated/torch.Generator.html) to make
                generation deterministic.
            latents (`torch.FloatTensor`, *optional*):
                Pre-generated noisy latents sampled from a Gaussian distribution, to be used as inputs for image
                generation. Can be used to tweak the same generation with different prompts. If not provided, a latents
                tensor is generated by sampling using the supplied random `generator`.
            output_type (`str`, *optional*, defaults to `"pil"`):
                The output format of the generated image. Choose between `PIL.Image` or `np.array`.
            callback_on_step_end (`Callable`, *optional*):
                A function that calls at the end of each denoising steps during the inference. The function is called
                with the following arguments: `callback_on_step_end(self: DiffusionPipeline, step: int, timestep: int,
                callback_kwargs: Dict)`. `callback_kwargs` will include a list of all tensors as specified by
                `callback_on_step_end_tensor_inputs`.
            callback_on_step_end_tensor_inputs (`List`, *optional*):
                The list of tensor inputs for the `callback_on_step_end` function. The tensors specified in the list
                will be passed as `callback_kwargs` argument. You will only be able to include variables listed in the
                `._callback_tensor_inputs` attribute of your pipeline class.
            return_dict (`bool`, *optional*, defaults to `True`):
                Whether or not to return a [`~pipelines.stable_diffusion.StableDiffusionPipelineOutput`] instead of a
                plain tuple.

        Returns:
            [`~pipelines.stable_diffusion.StableVideoDiffusionPipelineOutput`] or `tuple`:
                If `return_dict` is `True`, [`~pipelines.stable_diffusion.StableVideoDiffusionPipelineOutput`] is returned,
                otherwise a `tuple` is returned where the first element is a list of list with the generated frames.

        Examples:

        ```py
        from diffusers import StableVideoDiffusionPipeline
        from diffusers.utils import load_image, export_to_video

        pipe = StableVideoDiffusionPipeline.from_pretrained("stabilityai/stable-video-diffusion-img2vid-xt", torch_dtype=torch.float16, variant="fp16")
        pipe.to("cuda")

        image = load_image("https://lh3.googleusercontent.com/y-iFOHfLTwkuQSUegpwDdgKmOjRSTvPxat63dQLB25xkTs4lhIbRUFeNBWZzYf370g=s1200")
        image = image.resize((1024, 576))

        frames = pipe(image, num_frames=25, decode_chunk_size=8).frames[0]
        export_to_video(frames, "generated.mp4", fps=7)
        ```
        """
        # 0. Default height and width to unet
        height = height or self.unet.config.sample_size * self.vae_scale_factor
        width = width or self.unet.config.sample_size * self.vae_scale_factor

        num_frames = num_frames if num_frames is not None else self.unet.config.num_frames
        decode_chunk_size = decode_chunk_size if decode_chunk_size is not None else num_frames

        # 1. Check inputs. Raise error if not correct
        self.check_inputs(image, height, width)

        # 2. Define call parameters
        if isinstance(image, PIL.Image.Image):
            batch_size = 1
        elif isinstance(image, list):
            batch_size = len(image)
        else:
            batch_size = image.shape[0]
        device = self._execution_device
        # here `guidance_scale` is defined analog to the guidance weight `w` of equation (2)
        # of the Imagen paper: https://arxiv.org/pdf/2205.11487.pdf . `guidance_scale = 1`
        # corresponds to doing no classifier free guidance.
        do_classifier_free_guidance = max_guidance_scale > 1.0

        # 3. Encode input image
        if condition_type=="image":
            image_embeddings = self._encode_image(image, device, num_videos_per_prompt, do_classifier_free_guidance)
        elif condition_type=="text":
            if do_classifier_free_guidance:
                prompt_embeds = torch.cat([negative_prompt_embeds, prompt_embeds])
            image_embeddings = prompt_embeds
        else:
            image_embeddings = self._encode_image(image, device, num_videos_per_prompt, do_classifier_free_guidance)
            if do_classifier_free_guidance:
                prompt_embeds = torch.cat([negative_prompt_embeds, prompt_embeds])
            image_embeddings = torch.cat([image_embeddings, prompt_embeds], dim=1)
        motion_mask = self.unet.config.in_channels == 9
        if do_classifier_free_guidance:
            mask = torch.cat([mask]*2)
        # NOTE: Stable Diffusion Video was conditioned on fps - 1, which
        # is why it is reduced here.
        # See: https://github.com/Stability-AI/generative-models/blob/ed0997173f98eaf8f4edf7ba5fe8f15c6b877fd3/scripts/sampling/simple_video_sample.py#L188
        fps = fps - 1

        # 4. Encode input image using VAE
        image = self.image_processor.preprocess(image, height=height, width=width)
        noise = randn_tensor(image.shape, generator=generator, device=image.device, dtype=image.dtype)
        image = image + noise_aug_strength * noise

        needs_upcasting = self.vae.dtype == torch.float16 and self.vae.config.force_upcast
        if needs_upcasting:
            self.vae.to(dtype=torch.float32)

        if condition_latent is None:
            image_latents = self._encode_vae_image(image, device, num_videos_per_prompt, do_classifier_free_guidance)
            image_latents = image_latents.to(image_embeddings.dtype)

            # Repeat the image latents for each frame so we can concatenate them with the noise
            # image_latents [batch, channels, height, width] ->[batch, num_frames, channels, height, width]
            condition_latent = image_latents.unsqueeze(1).repeat(1, num_frames, 1, 1, 1)
        else:
            if do_classifier_free_guidance:
                condition_latent = torch.cat([condition_latent] * 2)
        # 5. Get Added Time IDs

        # cast back to fp16 if needed
        if needs_upcasting:
            self.vae.to(dtype=torch.float16)

        added_time_ids = self._get_add_time_ids(
            fps,
            motion_bucket_id,
            noise_aug_strength,
            image_embeddings.dtype,
            batch_size,
            num_videos_per_prompt,
            do_classifier_free_guidance,
        )
        added_time_ids = added_time_ids.to(device)

        # 4. Prepare timesteps
        self.scheduler.set_timesteps(num_inference_steps, device=device)
        timesteps = self.scheduler.timesteps

        # 5. Prepare latent variables
        num_channels_latents = self.unet.config.in_channels
        latents = self.prepare_latents(
            batch_size * num_videos_per_prompt,
            num_frames,
            num_channels_latents,
            height,
            width,
            image_embeddings.dtype,
            device,
            generator,
            latents,
        )

        # 7. Prepare guidance scale
        guidance_scale = torch.linspace(min_guidance_scale, max_guidance_scale, num_frames).unsqueeze(0)
        guidance_scale = guidance_scale.to(device, latents.dtype)
        guidance_scale = guidance_scale.repeat(batch_size * num_videos_per_prompt, 1)
        guidance_scale = _append_dims(guidance_scale, latents.ndim)

        self._guidance_scale = guidance_scale

        # 8. Denoising loop
        num_warmup_steps = len(timesteps) - num_inference_steps * self.scheduler.order
        self._num_timesteps = len(timesteps)
        with self.progress_bar(total=num_inference_steps) as progress_bar:
            for i, t in enumerate(timesteps):
                # expand the latents if we are doing classifier free guidance
                latent_model_input = torch.cat([latents] * 2) if do_classifier_free_guidance else latents
                latent_model_input = self.scheduler.scale_model_input(latent_model_input, t)

                if motion_mask:
                    # Concatenate image_latents over channels dimention
                    latent_model_input = torch.cat([mask, latent_model_input, condition_latent], dim=2)
                else:
                    latent_model_input = torch.cat([latent_model_input, condition_latent], dim=2)
                # predict the noise residual
                noise_pred = self.unet(
                    latent_model_input,
                    t,
                    encoder_hidden_states=image_embeddings,
                    added_time_ids=added_time_ids,
                    return_dict=False,
                )[0]

                # perform guidance
                if do_classifier_free_guidance:
                    noise_pred_uncond, noise_pred_cond = noise_pred.chunk(2)
                    noise_pred = noise_pred_uncond + self.guidance_scale * (noise_pred_cond - noise_pred_uncond)
                # compute the previous noisy sample x_t -> x_t-1
                latents = self.scheduler.step(noise_pred, t, latents).prev_sample
                if callback_on_step_end is not None:
                    callback_kwargs = {}
                    for k in callback_on_step_end_tensor_inputs:
                        callback_kwargs[k] = locals()[k]
                    callback_outputs = callback_on_step_end(self, i, t, callback_kwargs)

                    latents = callback_outputs.pop("latents", latents)

                if i == len(timesteps) - 1 or ((i + 1) > num_warmup_steps and (i + 1) % self.scheduler.order == 0):
                    progress_bar.update()

        if not output_type == "latent":
            # cast back to fp16 if needed
            if needs_upcasting:
                self.vae.to(dtype=torch.float16)
            frames = self.decode_latents(latents, num_frames, decode_chunk_size)
            frames = svd_tensor2vid(frames, self.image_processor, output_type=output_type)
        else:
            frames = latents

        self.maybe_free_model_hooks()

        if not return_dict:
            return frames

        return StableVideoDiffusionPipelineOutput(frames=frames)