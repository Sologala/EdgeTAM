import copy
import os
import gradio as gr
import cv2
import numpy as np
from datetime import datetime
import torch
from sam2.build_sam import build_sam2_video_predictor
from PIL import Image
import matplotlib.pyplot as plt


os.environ["TORCH_CUDNN_SDPA_ENABLED"] = "0,1,2,3,4,5,6,7"

OBJ_ID = 0

# 初始化全局变量
total_frames = 0
video_path = ''
frame_idx = 0
is_initialized = False
selected_points = {'inner': [], 'outer': []}  # 用于存储选择的点

# 初始化torch backend
if torch.backends.mps.is_available():
    DEVICE = "mps"
elif torch.cuda.is_available():
    DEVICE = "cuda"
else:
    DEVICE = "cpu"

# build sam model
sam2_checkpoint = "checkpoints/edgetam.pt"
model_cfg = "edgetam.yaml"
predictor = build_sam2_video_predictor(
    model_cfg, sam2_checkpoint, device=DEVICE)
print("PREDICTOR LOADED")

# use bfloat16 for the entire notebook
if torch.cuda.is_available():
    torch.autocast(device_type=DEVICE, dtype=torch.bfloat16).__enter__()
    if torch.cuda.get_device_properties(0).major >= 8:
        # turn on tfloat32 for Ampere GPUs (https://pytorch.org/docs/stable/notes/cuda.html#tensorfloat-32-tf32-on-ampere-devices)
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True


# 根据进度条获取视频帧


def get_frame(progress, session_state):

    if progress is None:
        return None  # 或者返回一个默认的帧或错误信息

    if session_state["cap"] is None:
        print("no cap")
        return None
    cap = session_state["cap"]

    frame_idx = int(progress * session_state["total_frame"])

    cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
    ret, frame = cap.read()

    if not ret:
        return None

    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)  # 转换为RGB格式
    session_state["cur_frame"] = frame
    return frame


# 初始化按钮逻辑
def init_tracking(session_state):

    session_state["inference_state"] = predictor.init_state(video_path=video_path)

    return session_state["cur_frame"]


# 使用 SAM 进行跟踪（示例，用假数据）


def preprocess_video_in(video_path, session_state):
    print("load bottom clicked", video_path)
    if video_path is None:
        session_state["cap"] = None
        session_state["total_frame"] = 0
        session_state["frameidx"] = 0
        return (
            gr.update(value=None),  # video info
            session_state
        )

    if session_state["cap"]:
        session_state["cap"].release()

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print("Error: Could not open video.")
        return (
            gr.update(value=None),  # output_video
            session_state
        )

    session_state["cap"] = cap
    session_state["video_path"] = video_path
    session_state["total_frame"] = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    session_state["fps"] = int(cap.get(cv2.CAP_PROP_FPS))
    session_state["frame_width"] = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    session_state["frame_height"] = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    session_state["input_points"] = []
    session_state["input_labels"] = []

    return [
        gr.update(
            value=f'视频加载成功！\n视频路径: {video_path}\n总帧数: {session_state["total_frame"]}\n分辨率: {session_state["frame_width"]}x{session_state["frame_height"]}\n帧率: {session_state["fps"]} FPS'),
        session_state
    ]

def show_mask(mask, obj_id=None, random_color=False, convert_to_image=True):
    if random_color:
        color = np.concatenate([np.random.random(3), np.array([0.6])], axis=0)
    else:
        cmap = plt.get_cmap("tab10")
        cmap_idx = 0 if obj_id is None else obj_id
        color = np.array([*cmap(cmap_idx)[:3], 0.6])
    h, w = mask.shape[-2:]
    mask = mask.reshape(h, w, 1) * color.reshape(1, 1, -1)
    mask = (mask * 255).astype(np.uint8)
    if convert_to_image:
        mask = Image.fromarray(mask, "RGBA")
    return mask



def segment_with_points(
    point_type,
    session_state,
    evt: gr.SelectData,
):
    session_state["input_points"].append(evt.index)
    print(f"TRACKING INPUT POINT: {session_state['input_points']}")

    if point_type == "include":
        session_state["input_labels"].append(1)
    elif point_type == "exclude":
        session_state["input_labels"].append(0)
    print(f"TRACKING INPUT LABEL: {session_state['input_labels']}")

    # Open the image and get its dimensions
    transparent_background = Image.fromarray(session_state["cur_frame"]).convert(
        "RGBA"
    )
    w, h = transparent_background.size

    # Define the circle radius as a fraction of the smaller dimension
    fraction = 0.01  # You can adjust this value as needed
    radius = int(fraction * min(w, h))

    # Create a transparent layer to draw on
    transparent_layer = np.zeros((h, w, 4), dtype=np.uint8)

    for index, track in enumerate(session_state["input_points"]):
        if session_state["input_labels"][index] == 1:
            cv2.circle(transparent_layer, track, radius, (0, 255, 0, 255), -1)
        else:
            cv2.circle(transparent_layer, track, radius, (255, 0, 0, 255), -1)

    # Convert the transparent layer back to an image
    transparent_layer = Image.fromarray(transparent_layer, "RGBA")
    selected_point_map = Image.alpha_composite(
        transparent_background, transparent_layer
    )

    # Let's add a positive click at (x, y) = (210, 350) to get started
    points = np.array(session_state["input_points"], dtype=np.float32)
    # for labels, `1` means positive click and `0` means negative click
    labels = np.array(session_state["input_labels"], np.int32)
    _, _, out_mask_logits = predictor.add_new_points(
        inference_state=session_state["inference_state"],
        frame_idx=0,
        obj_id=OBJ_ID,
        points=points,
        labels=labels,
    )

    mask_image = show_mask((out_mask_logits[0] > 0.0).cpu().numpy())
    first_frame_output = Image.alpha_composite(
        transparent_background, mask_image)

    torch.cuda.empty_cache()
    return selected_point_map, first_frame_output, session_state


def create_gradio_interface():

    with gr.Blocks() as demo:
        session_state = gr.State({"video_path": None, "cap": None,
                                  "frameidx": 0,
                                  "total_frame": 0,
                                  "fps": 0,
                                  "frame_width": 0,
                                  "frame_height": 0,
                                  "input_points": [],
                                  "input_labels": [],
                                  "inference_state": None})

        title = "<center><strong><font size='8'>EdgeTAM<font></strong> <a href='https://github.com/facebookresearch/EdgeTAM'><font size='6'>[GitHub]</font></a> </center>"
        gr.Markdown(title)

        video_input = gr.Textbox(
            label="Video URL or Path", value="/home/wen/Documents/1_1_大酱无人机10分钟，汽车跟随模式_10分钟不间断，大疆无人机的自动跟随到底靠谱吗？会不会撞，跟随是否流畅，飞行距离有多远。.mp4")
        # 显示视频基本信息
        video_info_output = gr.Textbox(label="Video Info", interactive=False)

        with gr.Row():
            # 视频路径输入框和加载按钮
            load_button = gr.Button("Load")

            point_type = gr.Radio(
                label="point type",
                choices=["include", "exclude"],
                value="include",
                scale=2,
            )
            init_btn = gr.Button("Init", scale=1, variant="primary")
            propagate_btn = gr.Button("Track", scale=1, variant="primary")
            clear_points_btn = gr.Button("Clear Points", scale=1)
            reset_btn = gr.Button("Reset", scale=1)

        # 进度条和显示视频帧
        progress_slider = gr.Slider(
            minimum=0, maximum=1, step=0.01, label="Progress")
        with gr.Row():
            video_frame_output = gr.Image(label="Current Video Frame")
            mask_image = gr.Image(label="Reference Mask")

        # 初始化和跟踪按钮
        track_button = gr.Button("Track")
        tracked_video_output = gr.Video(label="Tracked Video")

        # 将功能连接到回调函数
        load_button.click(fn=preprocess_video_in,
                          inputs=[video_input, session_state],
                          outputs=[video_info_output, session_state],
                          queue=False)

        progress_slider.change(
            fn=get_frame, inputs=[progress_slider, session_state], outputs=video_frame_output)

        init_btn.click(init_tracking, inputs=session_state,
                       outputs=mask_image)

        mask_image.select(
            fn=segment_with_points,
            inputs=[
                point_type,  # "include" or "exclude"
                session_state,
            ],
            outputs=[
                video_frame_output,  # updated image with points
                mask_image,
                session_state,
            ],
            queue=False,
        )

        # track_button.click(track_frames, outputs=tracked_video_output)

    return demo

# 启动应用


def main():
    demo = create_gradio_interface()
    demo.launch()


if __name__ == "__main__":
    main()
