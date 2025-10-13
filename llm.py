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

    def build_irreversible_events_prompt(self):
        file_ie_path = self.current_dir / "prompts" / "irreversible.txt"
        ie_prompt = utils.read_text_file(file_ie_path)
        self.input_ie_prompt = ie_prompt
        file_ie_image_path = self.current_dir / "images" / "task_camera_render.png"
        self.input_image_ie = utils.to_data_url(file_ie_image_path)

    def generate_irreversible_events(self):
        self.build_irreversible_events_prompt()
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
        return self.output_ie
    
    def build_reward_prompt(self):
        file_task_description_path = self.current_dir / "prompts" / "reward.txt"
        reward_prompt = utils.read_text_file(file_task_description_path)
        reward_prompt += "\n\n" + self.output_ie
        file_environment_class_path = self.current_dir / "env.py"
        env_class_text = utils.read_text_file(file_environment_class_path)
        reward_prompt += "\n\n# Environment class: \n" + env_class_text
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
        preference_prompt = utils.read_text_file(file_task_description_path)
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
        
    def generate_preference_from_video(self):
        input_content = [
        {"type": "input_text",
         "text": f"You will see several robot videos. TASK: The robot gripper is close to a cube, first it must reach the cube and and then with both the gripper fingers touch it and push it to the left.  Compare how successful the robot was in each episode\n"
                 "Each group of frames belongs to a different video."}
        ]
        
        video_path0 ="/home/leojellypc/cube_push/videos/lift_demo.mp4"
        video_path1 = "/home/leojellypc/cube_push/videos/lift_demo1.mp4"
        video_paths = [video_path0, video_path1]

        for idx, path in enumerate(video_paths, 1):
            b64_frames = utils.frame_sampler(path, every_n_frames=20, max_frames=16)
            input_content.append({"type": "input_text", "text": f"Video {idx}:"})
            for b in b64_frames:
                input_content.append({
                    "type": "input_image",
                    "image_url": f"data:image/jpeg;base64,{b}"
                })

        resp = self.client.responses.create(
            model=self.model,
            input=[{"role": "user", "content": input_content}],
        )
        print(resp.output_text)