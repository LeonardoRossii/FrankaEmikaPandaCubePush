import re
import utils
from openai import OpenAI
import importlib
import spec
from pathlib import Path
import inspect
import base64
import mimetypes

class GPT:
    def __init__(
        self,
        client: OpenAI,
        model: str = "gpt-5",
        temperature: float = 0.0,
        current_dir = Path(__file__).parent,
    ):
        self.model = model
        self.client = client
        self.temperature = temperature
        self.current_dir = current_dir
        self.input_prompt_reward = None
        self.input_prompt_prefer = None
        self.output_reward_path = None
        self.input_ie_prompt = None
        self.input_image_ie = None
        self.output_ie = None
                
    def build_reward_prompt(self):
        file_task_description_path = self.current_dir / "prompts" / "reward.txt"

        with open(file_task_description_path, "r") as file:
            reward_prompt = file.read().strip()

        reward_prompt += "\n\n" + self.output_ie

        file_environment_class_path = self.current_dir / "env.py"
        with open(file_environment_class_path, "r") as file:
            env_class = file.read().strip()
        reward_prompt += "\n\n# Environment class: \n" + env_class
        print(reward_prompt)
        self.input_prompt_reward = reward_prompt

        
    def generate_reward(self):
        self.build_reward_prompt()
        try:
            response = self.client.responses.create(
                model=self.model,
                input=self.input_prompt_reward,
                )
        except Exception as err:
            raise RuntimeError(f"OpenAI call failed: {err}")
        
        content = response.output_text
        content = utils.strip_code(content)
        
        with open(Path(inspect.getfile(spec)), "w") as f:
            f.write(content)
        importlib.reload(spec)
    
    def build_preference_prompt(self):
        file_task_description_path = self.current_dir / "prompts" / "feedback.txt"
        with open(file_task_description_path, "r") as file:
            preference_prompt = file.read().strip()
        self.input_prompt_prefer = preference_prompt

    def generate_preference(self, trajdesc):
        self.build_preference_prompt()
        try:
            response = self.client.responses.create(
                model=self.model,
                input=f"{self.input_prompt_prefer}\n\nTrajectories:\n{trajdesc}",
            )
        except Exception as err:
            raise RuntimeError(f"OpenAI call failed: {err}")

        answ = response.output_text
        match = re.search(r"idx\s*=\s*(\d+)", answ)
        if match:
            idx = int(match.group(1))
            return idx
        
    def build_ie_prompt(self):
        file_ie_path = self.current_dir / "prompts" / "irreversible.txt"
        with open(file_ie_path, "r") as file:
            ie_prompt = file.read().strip()
        self.input_ie_prompt = ie_prompt

    @staticmethod
    def _to_data_url(path: str) -> str:
        p = Path(path)
        mime, _ = mimetypes.guess_type(p.name)
        if mime is None:
            mime = "image/png"
        b64 = base64.b64encode(p.read_bytes()).decode("utf-8")
        return f"data:{mime};base64,{b64}"

    def build_ie_image(self):
        file_ie_image_path = self.current_dir / "images" / "task_camera_render.png"
        self.input_image_ie = self._to_data_url(file_ie_image_path)

    def generate_ie_with_image(self):
        self.build_ie_prompt()
        self.build_ie_image()
        try:
            response = self.client.responses.create(
                model=self.model,
                input=[{
                    "role": "user",
                    "content": [
                        {"type": "input_text", "text": self.input_ie_prompt},
                        {"type": "input_image", "image_url": self.input_image_ie},
                    ],
                }],
            )
        except Exception as err:
            raise RuntimeError(f"OpenAI call failed: {err}")
        self.output_ie = response.output_text 
