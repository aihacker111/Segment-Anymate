import gradio as gr
import os

from model_args import sam_args
from modules.segment import SegMent
import cv2
import numpy as np
import warnings

warnings.filterwarnings('ignore')


class SamModule:
    def __init__(self):
        pass

    @staticmethod
    def clean():
        return None, None, None, None, None, [[]]

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
        with open(temp_file_wrapper.name, 'rb') as f:
            # Read the content of the file
            file_content = f.read()
        return file_content

    def get_meta_from_image(self, input_img):
        file_content = self.read_temp_file(input_img)
        np_arr = np.frombuffer(file_content, np.uint8)

        img = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
        first_frame = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        return first_frame, first_frame

    @staticmethod
    def init_segment(points_per_side, origin_frame):
        if origin_frame is None:
            return None, origin_frame, [[], []]

        sam_args["generator_args"]["points_per_side"] = points_per_side

        segment = SegMent(sam_args)

        return segment, origin_frame, [[], []]

    @staticmethod
    def seg_acc_click(seg_tracker, prompt, origin_frame):
        # seg acc to click
        refined_mask, masked_frame = seg_tracker.seg_acc_click(
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

        masked_frame = self.seg_acc_click(segment, click_prompt, origin_frame)

        return segment, masked_frame, click_stack

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
            with gr.Tab(label='Image example'):
                gr.Examples(
                    examples=[
                        os.path.join(os.path.dirname(__file__), "../assets", "gradio.jpg"),
                    ],
                    inputs=[input_image],
                )

        app.queue(concurrency_count=1)
        app.launch(debug=True, share=True)


if __name__ == '__main__':
    run = SamModule()
    run.launch()
