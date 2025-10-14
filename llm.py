import re
import utils
from openai import OpenAI
import importlib
import spec
from pathlib import Path
import inspect
from env import Push
import agent

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
        #file_environment_class_path = self.current_dir / "env.py"
        #env_class_text = utils.read_text_file(file_environment_class_path)
        self.input_ie_prompt = ie_prompt
        #self.input_ie_prompt = ie_prompt + "\n\n# Environment class: \n" + env_class_text
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

    def build_preference_setup_prompt(self):
        file_task_description_path = self.current_dir / "prompts" / "feedback.txt"
        preference_prompt = utils.read_text_file(file_task_description_path)
        if self.output_ie is not None: preference_prompt += f"\n\n{self.output_ie}"
        preference_prompt += f"\nOBSERVABLES:\n\n{utils.extract_function_from_class(Push, '_setup_observables')}"
        file_agent_class_path = self.current_dir / "agent.py"
        env_agent_text = utils.read_text_file(file_agent_class_path)
        preference_prompt += env_agent_text
        self.input_prompt_prefer = preference_prompt

    def generate_preference_setup(self):
        self.build_preference_setup_prompt()
        try:
            response = self.client.responses.create(
                model=self.model,
                input= self.input_prompt_prefer,
            )
        except Exception as err:
            raise RuntimeError(f"OpenAI call failed: {err}")
        content = response.output_text
        with open(Path(inspect.getfile(agent)), "w") as f:
            f.write(content)
        importlib.reload(spec)
        
    def generate_preference(self):
        input_text = f"You will recive videos. The desired task is: The robot gripper is close to a cube, first it must reach the cube and and then with both the gripper fingers touch it and push it to the left.  Compare how successful the robot was in each episode based on the videos (Each group of frames belongs to a different video). At the end return me an the index corresponding to the best videos in this way: idx=0 or idx= 1 or idx=2."
        input_content = [
        {"type": "input_text",
         "text": input_text}
        ]
        
        video_path0 = "/home/leojellypc/cube_push/videos/video0.mp4"
        video_path1 = "/home/leojellypc/cube_push/videos/video1.mp4"
        video_path2 = "/home/leojellypc/cube_push/videos/video2.mp4"

        video_paths = [video_path0, video_path1, video_path2]

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
        answ = resp.output_text
        print(answ)
        match = re.search(r"idx\s*=\s*(\d+)", answ)
        if match:
            idx = int(match.group(1))
        return idx