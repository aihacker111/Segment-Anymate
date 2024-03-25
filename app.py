from animation import generate_sam_args, SegMent, GenerativeMotion
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
        pass

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
    def download_models(mode):
        dir_path = os.path.join(os.getcwd(), 'root_model')
        models_urls = {
            'ld_models': 'https://cloudbook-public-production.oss-cn-shanghai.aliyuncs.com/animation/animate_anything_512_v1.02.tar',
            'sam_models': 'https://huggingface.co/ybelkada/segment-anything/resolve/main/checkpoints/sam_vit_b_01ec64.pth?download=true'
        }
        if not os.path.exists(dir_path):
            subdirectories = ['ld_models', 'sam_models']
            subdirectory_paths = [os.makedirs(os.path.join(dir_path, sub), exist_ok=True) or os.path.join(dir_path, sub)
                                  for
                                  sub in subdirectories]
            sam_models_path = subdirectory_paths[subdirectories.index('sam_models')]
            ld_models_path = subdirectory_paths[subdirectories.index('ld_models')]
            sam_models_name = os.path.join(sam_models_path, 'sam_vit_b.pth')
            ld_models_name = os.path.join(ld_models_path, 'ld_models.tar')

            print("Downloading SD models...")
            gdown.download(models_urls['sam_models'], sam_models_name, quiet=False)
            gdown.download(models_urls['ld_models'], ld_models_name, quiet=False)
            with tarfile.open(ld_models_name, 'r') as tar:
                tar.extractall(path=ld_models_path)
            print("Extraction complete.")
            if os.path.exists(ld_models_name):
                shutil.rmtree(ld_models_name)
            return "Download complete."
        else:
            return "Models are downloaded"

    def get_models_path(self):
        sam_models_path = os.path.join(os.getcwd(), 'root_model', 'sam_models')
        ld_models_path = os.path.join(os.getcwd(), 'root_model', 'ld_models', 'animate_anything_512_v1.02')
        sam_args = generate_sam_args(sam_checkpoint=sam_models_path)
        return sam_args, ld_models_path

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

    def init_segment(self, points_per_side, origin_frame):
        if origin_frame is None:
            return None, origin_frame, [[], []]
        sam_args, _ = self.get_models_path()
        sam_args["generator_args"]["points_per_side"] = points_per_side

        segment = SegMent(sam_args)

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

    def undo_click_stack_and_refine_seg(self, seg_tracker, origin_frame, click_stack):
        if seg_tracker is None:
            return seg_tracker, origin_frame, [[], []]

        print("Undo!")
        if len(click_stack[0]) > 0:
            click_stack[0] = click_stack[0][: -1]
            click_stack[1] = click_stack[1][: -1]

        if len(click_stack[0]) > 0:
            prompt = {
                "points_coord": click_stack[0],
                "points_mode": click_stack[1],
                "multi_mask": "True",
            }

            masked_frame = self.seg_acc_click(seg_tracker, prompt, origin_frame)
            return seg_tracker, masked_frame, click_stack
        else:
            return seg_tracker, origin_frame, [[], []]

    def sam_click(self, segment, origin_frame, point_mode, click_stack, long_term_mem, max_len_long_term,
                  evt: gr.SelectData):
        print("Click")

        if point_mode == "Positive":
            point = {"coord": [evt.index[0], evt.index[1]], "mode": 1}
        else:
            point = {"coord": [evt.index[0], evt.index[1]], "mode": 0}

        if segment is None:
            segment, _, _ = self.init_segment(long_term_mem, max_len_long_term)

        click_prompt = self.get_click_prompt(click_stack, point)

        refined_mask, masked_frame = self.seg_acc_click(segment, click_prompt, origin_frame)
        self.save_mask(refined_mask, save=True)
        return segment, masked_frame, click_stack

    def run(self, image, text):
        _, img_name = self.read_temp_file(image)
        _, pretrained_models_path = self.get_models_path()
        generative_motion = GenerativeMotion(pretrained_model_path=pretrained_models_path, prompt_image=img_name, prompt=text,
                                             mask=self.save_mask(refined_mask=None))
        generative_motion.render()
        return 'Rendering...'


class AnimateLaunch(AnimateController):
    def __init__(self):
        super().__init__()

    def launch(self):
        app = gr.Blocks()

        with app:
            gr.Markdown(
                '''
                    <div style="text-align:center;">
                        <span style="font-size:3em; font-weight:bold;">AI Generative Motion </span>
                    </div>
                    '''
            )

            click_stack = gr.State([[], []])  # Storage clicks status
            origin_frame = gr.State(None)
            segment = gr.State(None)
            render_status = gr.Textbox(label='Render Status')
            prompt_text = gr.Textbox(label='Text Prompt')
            models_download = gr.Textbox(label='Models Download Status')
            with gr.Row():
                with gr.Column(scale=1):
                    tab_image_input = gr.Tab(label="Image type input")
                    with tab_image_input:
                        input_image = gr.File(label='Input image')

                    input_first_frame = gr.Image(label='Segment result', interactive=True)
                    input_first_frame.style(height=550)
                    tab_click = gr.Tab(label="Click")
                    with tab_click:
                        with gr.Row():
                            point_mode = gr.Radio(
                                choices=["Positive", "Negative"],
                                value="Positive",
                                label="Point Prompt",
                                interactive=True)

                            click_undo_but = gr.Button(
                                value="Undo",
                                interactive=True
                            )
                            click_render = gr.Button(
                                value='Render',
                                interactive=True
                            )
                            click_download = gr.Button(
                                value='Download Models',
                                interactive=True
                            )
                            with gr.Accordion("aot advanced options", open=False):
                                long_term_mem = gr.Slider(label="long term memory gap", minimum=1, maximum=9999,
                                                          value=9999, step=1)
                                max_len_long_term = gr.Slider(label="max len of long term memory", minimum=1,
                                                              maximum=9999, value=9999, step=1)

            input_image.change(
                fn=self.get_meta_from_image,
                inputs=[
                    input_image
                ],
                outputs=[
                    input_first_frame, origin_frame
                ]
            )

            # -------------- Input com_pont -------------
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

            # ------------------- Interactive component -----------------

            tab_click.select(
                fn=self.init_segment,
                inputs=[
                    long_term_mem, origin_frame
                ],
                outputs=[
                    segment, input_first_frame, click_stack
                ],
                queue=False,
            )
            input_first_frame.select(
                fn=self.sam_click,
                inputs=[
                    segment, origin_frame, point_mode, click_stack,
                    long_term_mem,
                    max_len_long_term,
                ],
                outputs=[
                    segment, input_first_frame, click_stack
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
                        prompt_text],
                outputs=[
                    render_status
                ]
            )
            click_download.click(
                fn=self.download_models,
                inputs=[models_download],
                outputs=[models_download]
            )
            with gr.Tab(label='Image example'):
                gr.Examples(
                    examples=[
                        os.path.join(os.path.dirname(__file__), "assets", "gradio.jpg"),
                    ],
                    inputs=[input_image],
                )

        app.queue(concurrency_count=1)
        app.launch(debug=True, share=True)


if __name__ == '__main__':
    render = AnimateLaunch()
    render.launch()
