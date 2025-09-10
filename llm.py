import os
import spec
import utils
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