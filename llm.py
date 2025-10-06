import re
import utils
from openai import OpenAI
import importlib
import spec
from pathlib import Path
import inspect

class GPT:
    def __init__(
        self,
        client: OpenAI,
        model: str = "gpt-4o",
        temperature: float = 0.0,
    ):
        self.client = client
        self.model = model
        self.temperature = temperature
        self.input_prompt_reward = None
        self.input_prompt_prefer = None
        self.output_reward_path = None

    def build_reward_prompt(self):
        current_dir = Path(__file__).parent
        file_task_description_path = current_dir / "prompts" / "reward.txt"
        with open(file_task_description_path, "r") as file:
            reward_prompt = file.read().strip()
        file_environment_class_path = current_dir / "env.py"
        with open(file_environment_class_path, "r") as file:
            env_class = file.read().strip()
        reward_prompt += "\n\n# Environment class: \n" + env_class
        self.input_prompt_reward = reward_prompt
        
    def generate_reward(self):
        self.build_reward_prompt()
        try:
            response = self.client.responses.create(
                model=self.model,
                input=self.input_prompt_reward,
                temperature=self.temperature,
            )
        except Exception as err:
            raise RuntimeError(f"OpenAI call failed: {err}")
        
        content = response.output_text
        content = utils.strip_code(content)
        
        with open(Path(inspect.getfile(spec)), "w") as f:
            f.write(content)
        importlib.reload(spec)
    
    def build_preference_prompt(self):
        current_dir = Path(__file__).parent
        file_task_description_path = current_dir / "prompts" / "feedback.txt"
        with open(file_task_description_path, "r") as file:
            preference_prompt = file.read().strip()
        self.input_prompt_prefer = preference_prompt

    def generate_preference(self, trajdesc):
        self.build_preference_prompt()
        try:
            response = self.client.responses.create(
                model=self.model,
                input=f"{self.input_prompt_prefer}\n\nTrajectories:\n{trajdesc}",
                temperature=self.temperature
            )
        except Exception as err:
            raise RuntimeError(f"OpenAI call failed: {err}")

        answ = response.output_text
        print(answ)
        match = re.search(r"idx\s*=\s*(\d+)", answ)
        if match:
            idx = int(match.group(1))
            return idx