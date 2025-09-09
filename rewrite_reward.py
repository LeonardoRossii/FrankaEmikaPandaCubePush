import os
import inspect
import reward
from pathlib import Path
from openai import OpenAI

def strip_code_fence(code_str):
    lines = code_str.strip().splitlines()
    if lines[0].startswith("```"):
        lines = lines[1:]
    if lines and lines[-1].startswith("```"):
        lines = lines[:-1]
    return "\n".join(lines)

def rewrite_reward(
    prompt: str,
    reward_file_path: str = Path(inspect.getfile(reward)),
    model: str = "gpt-4o",
    temperature: float = 0.0
    ):
    client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
    if not client.api_key:
        raise RuntimeError("OPENAI_API_KEY not set")
    try:
        response = client.responses.create(
            model=model,
            input=prompt,
            temperature=temperature,
        )
    except Exception as err:
        raise RuntimeError(f"OpenAI call failed: {err}")
    content = response.output_text
    content = strip_code_fence(content)
    with open(reward_file_path, "w") as f:
        f.write(content)