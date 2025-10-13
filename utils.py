import cv2
import os
import inspect
import numpy as np
import base64
import mimetypes
from pathlib import Path
from robosuite.environments import ALL_ENVIRONMENTS

def register_environment(env, name):
    if name not in ALL_ENVIRONMENTS: ALL_ENVIRONMENTS[name] = env

def strip_code(code_str):
    lines = code_str.strip().splitlines()
    if lines[0].startswith("```"):
        lines = lines[1:]
    if lines and lines[-1].startswith("```"):
        lines = lines[:-1]
    return "\n".join(lines)

def read_text_file(path: Path) -> str:
    with open(path, "r") as file:
        return file.read().strip()

def to_data_url(path: str) -> str:
    p = Path(path)
    mime, _ = mimetypes.guess_type(p.name)
    if mime is None:
        mime = "image/png"
    b64 = base64.b64encode(p.read_bytes()).decode("utf-8")
    return f"data:{mime};base64,{b64}"

def save_video(frames, filename="output.mp4", fps=20):
    if len(frames) == 0:
        raise ValueError("No frames to save!")
    filepath = os.path.join("videos", filename)
    h, w, _ = frames[0].shape
    out = cv2.VideoWriter(filepath, cv2.VideoWriter_fourcc(*"mp4v"), fps, (w, h))
    for frame in frames:
        out.write(np.uint8(frame[:, :, ::-1]))
    out.release()
    print(f"Video saved to {filename}")

def frame_sampler(
    video_path: str,
    every_n_frames: int = 20,
    max_frames: int = 40,    
    width: int = 640,        
    vflip: bool = False
):
    cap = cv2.VideoCapture(video_path)
    frames_b64 = []
    i = 0
    while cap.isOpened() and len(frames_b64) < max_frames:
        ok, frame = cap.read()
        if not ok:
            break
        if i % every_n_frames == 0:
            if vflip:
                frame = cv2.flip(frame, 0)
            h, w = frame.shape[:2]
            new_h = int(h * (width / w))
            frame = cv2.resize(frame, (width, new_h))
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            ok, jpg = cv2.imencode(".jpg", rgb, [int(cv2.IMWRITE_JPEG_QUALITY), 85])
            if ok:
                frames_b64.append(base64.b64encode(jpg.tobytes()).decode("utf-8"))
        i += 1
    cap.release()
    return frames_b64

def extract_function_from_class(cls, func_name):
    source = inspect.getsource(getattr(cls, func_name))
    return source