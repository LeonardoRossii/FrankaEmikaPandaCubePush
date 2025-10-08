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

