import os
import gradio as gr
import cv2


def stop():
    return gr.Image()


def display(input_image):
    img_path = save_image(input_image, is_path=True)
    img = cv2.imread(img_path)
    return img


def save_image(img_input, is_path=False):
    input_path = os.path.join(os.getcwd(), 'webcam_image')
    os.makedirs(input_path, exist_ok=True)
    image_path = os.path.join(input_path, 'webcam' + '.jpg')
    cv2.imwrite(image_path, img_input[:, :, ::-1])
    if is_path:
        return image_path


def btn(btn):
    if btn == 'Capture':
        image_input = gr.Image(source='webcam', streaming=True)
        return image_input


with gr.Blocks() as demo:
    image_input = gr.Image(source='webcam', streaming=True)
    status = gr.Textbox(label='Saved Status')
    capture = gr.Button("Capture")
    stops = gr.Button("Stop Webcam")
    image_output = gr.Image()
    capture.click(
        display,
        inputs=image_input,
        outputs=image_output
    )


demo.launch(share=True)
