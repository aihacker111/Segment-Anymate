from animation import generate_sam_args, SegMent, GenerativeMotion, logger
import numpy as np
import gradio as gr
import os
import cv2
import gdown
import shutil
import tarfile
import warnings

warnings.filterwarnings('ignore')


class AnimateController:
    def __init__(self):
        self.current_model_type = None

    @staticmethod
    def clean():
        return None, None, None, None, None, [[]]

    @staticmethod
    def save_mask(refined_mask, save=False):

        if save:
            if os.path.exists(os.path.join(os.getcwd(), 'output_render')):
                shutil.rmtree(os.path.join(os.getcwd(), 'output_render'))
            save_path = os.path.join(os.getcwd(), 'output_render')
            os.makedirs(save_path, exist_ok=True)
            cv2.imwrite(os.path.join(save_path, f'refined_mask_result.png'), refined_mask)
        else:
            return os.path.join(os.path.join(os.getcwd(), 'output_render'), f'refined_mask_result.png')

    @staticmethod
    def download_models(model_type):
        dir_path = os.path.join(os.getcwd(), 'root_model')
        ld_models_path = os.path.join(dir_path, 'ld_models')
        sam_models_path = os.path.join(dir_path, 'sam_models')
        # Models URLs
        models_urls = {
            'ld_models': {
                'latent_diffusion': 'https://cloudbook-public-production.oss-cn-shanghai.aliyuncs.com/animation/animate_anything_512_v1.02.tar'},
            'sam_models': {
                'vit_b': 'https://huggingface.co/ybelkada/segment-anything/resolve/main/checkpoints/sam_vit_b_01ec64.pth?download=true',
                'vit_l': 'https://huggingface.co/segments-arnaud/sam_vit_l/resolve/main/sam_vit_l_0b3195.pth?download=true',
                'vit_h': 'https://huggingface.co/segments-arnaud/sam_vit_h/resolve/main/sam_vit_h_4b8939.pth?download=true'
            }
        }
        if os.path.exists(ld_models_path) or model_type in models_urls['ld_models']:
            return 'Animate Models is already exists'
        # Check if ld_models exist, if not, download them
        if not os.path.exists(ld_models_path) and model_type in models_urls['ld_models']:
            os.makedirs(ld_models_path, exist_ok=True)
            ld_models_name = os.path.join(ld_models_path, 'ld_models.tar')
            logger.info("Downloading LD models...")
            gdown.download(models_urls['ld_models'][model_type], ld_models_name, quiet=False)
            with tarfile.open(ld_models_name, 'r') as tar:
                tar.extractall(path=ld_models_path)
            logger.info("Extraction complete.")
            os.remove(ld_models_name)
            return 'Animate Models is downloaded'

        # Download specified model type
        if model_type in models_urls['sam_models']:
            model_url = models_urls['sam_models'][model_type]
            os.makedirs(sam_models_path, exist_ok=True)
            model_path = os.path.join(sam_models_path, model_type + '.pth')
            if not os.path.exists(model_path):
                logger.info(f"Downloading {model_type} model...")
                gdown.download(model_url, model_path, quiet=False)
                logger.info(f"{model_type} model downloaded.")
            else:
                logger.info(f"{model_type} model already exists.")
            return logger.info(f"{model_type} model download complete.")
        else:
            return logger.info(f"Invalid model type: {model_type}")

    def get_models_path(self, model_type=None, segment=False, diffusion=False):
        sam_models_path = os.path.join(os.getcwd(), 'root_model', 'sam_models')
        ld_models_path = os.path.join(os.getcwd(), 'root_model', 'ld_models', 'animate_anything_512_v1.02')
        if segment:
            sam_args = generate_sam_args(sam_checkpoint=sam_models_path, model_type=model_type)
            return sam_args, sam_models_path
        elif diffusion:
            return ld_models_path

    @staticmethod
    def get_click_prompt(click_stack, point):
        click_stack[0].append(point["coord"])
        click_stack[1].append(point["mode"]
                              )

        prompt = {
            "points_coord": click_stack[0],
            "points_mode": click_stack[1],
            "multi_mask": "True",
        }

        return prompt

    @staticmethod
    def read_temp_file(temp_file_wrapper):
        name = temp_file_wrapper.name
        with open(temp_file_wrapper.name, 'rb') as f:
            # Read the content of the file
            file_content = f.read()
        return file_content, name

    def get_meta_from_image(self, input_img):
        file_content, _ = self.read_temp_file(input_img)
        np_arr = np.frombuffer(file_content, np.uint8)

        img = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
        first_frame = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        return first_frame, first_frame

    def is_sam_model(self, model_type):
        sam_args, sam_models_dir = self.get_models_path(model_type=model_type, segment=True)
        model_path = os.path.join(sam_models_dir, model_type + '.pth')
        if not os.path.exists(model_path):
            self.download_models(model_type=model_type)
            return 'Model is downloaded', sam_args
        else:
            return 'Model is already downloaded', sam_args

    def init_segment(self,
                     points_per_side,
                     origin_frame,
                     sam_args,
                     predict_iou_thresh=0.8,
                     stability_score_thresh=0.9,
                     crop_n_layers=1,
                     crop_n_points_downscale_factor=2,
                     min_mask_region_area=200):
        if origin_frame is None:
            return None, origin_frame, [[], []]
        sam_args["generator_args"]["points_per_side"] = points_per_side
        sam_args["generator_args"]["pred_iou_thresh"] = predict_iou_thresh
        sam_args["generator_args"]["stability_score_thresh"] = stability_score_thresh
        sam_args["generator_args"]["crop_n_layers"] = crop_n_layers
        sam_args["generator_args"]["crop_n_points_downscale_factor"] = crop_n_points_downscale_factor
        sam_args["generator_args"]["min_mask_region_area"] = min_mask_region_area

        segment = SegMent(sam_args)
        logger.info(f"Model Init: {sam_args}")
        return segment, origin_frame, [[], []]

    @staticmethod
    def seg_acc_click(segment, prompt, origin_frame):
        # seg acc to click
        refined_mask, masked_frame = segment.seg_acc_click(
            origin_frame=origin_frame,
            coords=np.array(prompt["points_coord"]),
            modes=np.array(prompt["points_mode"]),
            multimask=prompt["multi_mask"],
        )
        return refined_mask, masked_frame

    def undo_click_stack_and_refine_seg(self, segment, origin_frame, click_stack):
        if segment is None:
            return segment, origin_frame, [[], []]

        logger.info("Undo !")
        if len(click_stack[0]) > 0:
            click_stack[0] = click_stack[0][: -1]
            click_stack[1] = click_stack[1][: -1]

        if len(click_stack[0]) > 0:
            prompt = {
                "points_coord": click_stack[0],
                "points_mode": click_stack[1],
                "multi_mask": "True",
            }

            _, masked_frame = self.seg_acc_click(segment, prompt, origin_frame)
            return segment, masked_frame, click_stack
        else:
            return segment, origin_frame, [[], []]

    def reload_segment(self,
                       check_sam,
                       segment,
                       model_type,
                       point_per_sides,
                       origin_frame,
                       predict_iou_thresh,
                       stability_score_thresh,
                       crop_n_layers,
                       crop_n_points_downscale_factor,
                       min_mask_region_area):
        status, sam_args = check_sam(model_type)
        if segment is None or status == 'Model is downloaded':
            segment, _, _ = self.init_segment(point_per_sides,
                                              origin_frame,
                                              sam_args,
                                              predict_iou_thresh,
                                              stability_score_thresh,
                                              crop_n_layers,
                                              crop_n_points_downscale_factor,
                                              min_mask_region_area)
            self.current_model_type = model_type
        return segment, self.current_model_type, status

    def sam_click(self,
                  evt: gr.SelectData,
                  segment,
                  origin_frame,
                  model_type,
                  point_mode,
                  click_stack,
                  point_per_sides,
                  predict_iou_thresh,
                  stability_score_thresh,
                  crop_n_layers,
                  crop_n_points_downscale_factor,
                  min_mask_region_area):
        logger.info("Click")
        if point_mode == "Positive":
            point = {"coord": [evt.index[0], evt.index[1]], "mode": 1}
        else:
            point = {"coord": [evt.index[0], evt.index[1]], "mode": 0}
        click_prompt = self.get_click_prompt(click_stack, point)
        segment, self.current_model_type, status = self.reload_segment(
            self.is_sam_model,
            segment,
            model_type,
            point_per_sides,
            origin_frame,
            predict_iou_thresh,
            stability_score_thresh,
            crop_n_layers,
            crop_n_points_downscale_factor,
            min_mask_region_area)
        if segment is not None and model_type != self.current_model_type:
            segment = None
            segment, _, status = self.reload_segment(
                self.is_sam_model,
                segment,
                model_type,
                point_per_sides,
                origin_frame,
                predict_iou_thresh,
                stability_score_thresh,
                crop_n_layers,
                crop_n_points_downscale_factor,
                min_mask_region_area)
        refined_mask, masked_frame = self.seg_acc_click(segment, click_prompt, origin_frame)
        self.save_mask(refined_mask, save=True)
        return segment, masked_frame, click_stack, status

    def run(self, image, text, num_frames, num_inference_steps, guidance_scale, fps):
        _, img_name = self.read_temp_file(image)
        pretrained_models_path = self.get_models_path(model_type=None, diffusion=True)
        generative_motion = GenerativeMotion(pretrained_model_path=pretrained_models_path, prompt_image=img_name,
                                             prompt=text,
                                             mask=self.save_mask(refined_mask=None))
        final_vid_path = generative_motion.render(num_frames=num_frames, num_inference_steps=num_inference_steps,
                                                  guidance_scale=guidance_scale, fps=fps)
        return final_vid_path


class AnimateLaunch(AnimateController):
    def __init__(self):
        super().__init__()

    def launch(self):
        app = gr.Blocks()

        with app:
            gr.Markdown(
                '''
                    <div style="text-align:center;">
                        <span style="font-size:3em; font-weight:bold;">Segment AnyMate </span>
                    </div>
                    '''
            )

            click_stack = gr.State([[], []])  # Storage clicks status
            origin_frame = gr.State(None)
            segment = gr.State(None)
            with gr.Row():
                with gr.Column():
                    tab_image_input = gr.Tab(label="Upload Image")
                    with tab_image_input:
                        input_image = gr.File(label='Input image')

                with gr.Column():
                    tab_click = gr.Tab(label="Segment Anything Setting")
                    with tab_click:
                        with gr.Column():
                            model_type = gr.Radio(
                                choices=["vit_b", "vit_l", "vit_h", 'latent_diffusion'],
                                value="vit_b",
                                label="SAM Models Type",
                                interactive=True
                            )
                        with gr.Column():
                            point_mode = gr.Radio(
                                choices=["Positive", "Negative"],
                                value="Positive",
                                label="Point Prompt",
                                interactive=True)

                            click_undo_but = gr.Button(
                                value="Undo",
                                interactive=True
                            )
                        with gr.Column():
                            with gr.Accordion("SAM Advanced Options", open=True):
                                with gr.Row():
                                    point_per_side = gr.Slider(label="point per sides", minimum=1, maximum=100,
                                                               value=16, step=1)
                                    predict_iou_thresh = gr.Slider(label="IoU Threshold", minimum=0, maximum=1,
                                                                   value=0.8,
                                                                   step=0.1)
                                    score_thresh = gr.Slider(label="Scored Threshold", minimum=0, maximum=1, value=0.9,
                                                             step=0.1)
                                    crop_n_layers = gr.Slider(label="Crop Layers", minimum=0, maximum=100, value=1,
                                                              step=1)
                                    crop_n_points = gr.Slider(label="Crop Points", minimum=0, maximum=100, value=2,
                                                              step=1)
                                    min_mask_region_area = gr.Slider(label="Mask Region Area", minimum=0, maximum=1000,
                                                                     value=100, step=100)

                with gr.Column(scale=1):
                    models_download = gr.Textbox(label='Models Download Status')
                    with gr.Row():
                        input_first_frame = gr.Image(label='Segment Result', interactive=True).style(height=350)
                        output_video = gr.File(label="Predicted Video").style(height=350)
                    with gr.Row():
                        prompt_text = gr.Textbox(label='Text Prompt')
                    with gr.Row():
                        click_render = gr.Button(
                            value='Render',
                            interactive=True
                        )
                    tab_clicks = gr.Tab(label="Animate Setting")
                    with tab_clicks:
                        with gr.Accordion("Animate Advanced Options", open=True):
                            with gr.Row():
                                num_frames = gr.Slider(label="Number Of Frames", minimum=0, maximum=100, value=16,
                                                       step=16)
                                num_inference_steps = gr.Slider(label="Inference Steps", minimum=0, maximum=100, value=25,
                                                                step=1)
                                guidance_scale = gr.Slider(label="Guidance Scale", minimum=1, maximum=8, value=8,
                                                           step=1)
                                fps = gr.Slider(label="FPS", minimum=0, maximum=60,
                                                value=8, step=4)
                                download_animate_model = gr.Button(value="Download Animate Model",
                                                                   interactive=True)
                                animate_model_type = gr.Radio(
                                    choices=['latent_diffusion', 'animate_diff_motion'],
                                    value="vit_b",
                                    label="SAM Models Type",
                                    interactive=True
                                )
            input_image.change(
                fn=self.get_meta_from_image,
                inputs=[
                    input_image
                ],
                outputs=[
                    input_first_frame, origin_frame
                ]
            )

            tab_image_input.select(
                fn=self.clean,
                inputs=[],
                outputs=[
                    input_image,
                    segment,
                    input_first_frame,
                    origin_frame,
                    click_stack,
                ]
            )

            tab_click.select(
                fn=self.init_segment,
                inputs=[
                    point_per_side, origin_frame, model_type, predict_iou_thresh, score_thresh,
                    crop_n_layers, crop_n_points, min_mask_region_area
                ],
                outputs=[
                    segment, input_first_frame, click_stack
                ],
                queue=False,
            )
            input_first_frame.select(
                fn=self.sam_click,
                inputs=[
                    segment, origin_frame, model_type, point_mode, click_stack,
                    point_per_side, predict_iou_thresh, score_thresh,
                    crop_n_layers, crop_n_points, min_mask_region_area
                ],
                outputs=[
                    segment, input_first_frame, click_stack, models_download
                ]
            )

            click_undo_but.click(
                fn=self.undo_click_stack_and_refine_seg,
                inputs=[
                    segment, origin_frame, click_stack
                ],
                outputs=[
                    segment, input_first_frame, click_stack
                ]
            )
            click_render.click(
                fn=self.run,
                inputs=[input_image,
                        prompt_text,
                        num_frames,
                        num_inference_steps,
                        guidance_scale,
                        fps],
                outputs=[
                    output_video
                ]
            )
            download_animate_model.click(
                fn=self.download_models,
                inputs=[animate_model_type],
                outputs=[models_download]
            )

        app.queue(concurrency_count=1)
        app.launch(debug=True, share=True)


if __name__ == '__main__':
    render = AnimateLaunch()
    render.launch()
