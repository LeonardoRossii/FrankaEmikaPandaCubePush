import re
import os
import spec
import utils
import json
import inspect
from pathlib import Path
from openai import OpenAI

def generate_spec(
    prompt: str,
    reward_file_path: str = Path(inspect.getfile(spec)),
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
    content = utils.strip_code(content)
    
    with open(reward_file_path, "w") as f:
        f.write(content)


def get_preference(agent, weights, max_n_timesteps, prompt):

    client = OpenAI(api_key = os.getenv("OPENAI_API_KEY"))
    if not client.api_key:
        raise RuntimeError("OPENAI_API_KEY not set")

    trajs = [agent.episode(weight, max_n_timesteps) for weight in weights]
    tdesc = "\n\n".join(f"Trajectory {i}:\n{json.dumps(t, indent=2)}" for i, t in enumerate(trajs))
    prompt = f"{prompt.rstrip()}\n\nTrajectories:\n{tdesc}"
    try:
        response = client.responses.create(
            model = "gpt-4o",
            input = prompt,
            temperature = 0.0
        )
    except Exception as err:
        raise RuntimeError(f"OpenAI call failed: {err}")

    answ = response.output_text
    match = re.search(r"idx\s*=\s*(\d+)", answ)
    if match:
        idx = int(match.group(1))

    return idx