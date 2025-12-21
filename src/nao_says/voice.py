
import json
import re
import threading

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from RealtimeSTT import AudioToTextRecorder

class NaoVoiceCommand():
    def __init__(self, model_dir):
        # joint settings
        self.joints = ["HeadYaw","HeadPitch",
            "LShoulderPitch","LShoulderRoll","LElbowYaw","LElbowRoll","LWristYaw","LHand",
            "RShoulderPitch","RShoulderRoll","RElbowYaw","RElbowRoll","RWristYaw","RHand",
            "HipYawPitch","LHipRoll","LHipPitch","LKneePitch","LAnklePitch","LAnkleRoll",
            "RHipRoll","RHipPitch","RKneePitch","RAnklePitch","RAnkleRoll"]
        self.joints_angle_ranges = {"HeadYaw": (-119.5, 119.5), "HeadPitch": (-38.5, 29.5),
            "LShoulderPitch": (-119.5, 119.5), "LShoulderRoll": (-18.0, 76.0), "LElbowYaw": (-119.5, 119.5), "LElbowRoll": (-88.5, -2.0), "LWristYaw": (-104.5, 104.5), "LHand": (0.0, 1.0),
            "RShoulderPitch": (-119.5, 119.5), "RShoulderRoll": (-76.0, 18.0), "RElbowYaw": (-119.5, 119.5), "RElbowRoll": (2.0, 88.5), "RWristYaw": (-104.5, 104.5), "RHand": (0.0, 1.0),
            "HipYawPitch": (-65.62, 42.48), "LHipRoll": (-21.73, 45.0), "LHipPitch": (-88.0, 27.73), "LKneePitch": (0.0, 121.0), "LAnklePitch": (-68.0, 52.87), "LAnkleRoll": (-22.84, 44.5),
            "RHipRoll": (-45.0, 21.73), "RHipPitch": (-88.0, 27.73), "RKneePitch": (0.0, 121.0), "RAnklePitch": (-68.0, 52.87), "RAnkleRoll": (-44.5, 22.84),}

        # load audio recorder
        self.recorder = AudioToTextRecorder(language="en", device="cpu", compute_type="float32")
        self.last_cmd = None
        self._done = threading.Event()

        # load model and tokenizer
        self.model = AutoModelForCausalLM.from_pretrained(model_dir, local_files_only=True)
        self.tokenizer = AutoTokenizer.from_pretrained(model_dir, local_files_only=True)

    def close(self):
        if self.recorder:
            self.recorder.shutdown()
            self.recorder = None

    def record_audio(self):
        self._done.clear()
        self.recorder.text(self.process_audio)
        self._done.wait()
        
        return json.dumps(self.last_cmd, ensure_ascii=True)

    def process_audio(self, text):
        print("Recognized Text:", text)
        self.last_cmd = self.extract_command(text)
        self._done.set()

    def extract_command(self, text: str) -> dict:
        print("Extracting command from text...")
        prompt = f"""
You are a parser for NAO robot commands.
Output ONLY a single JSON object. No markdown. No extra text. No explanations.

Allowed keys (exactly these, no others):
wakeword, action, params

Schema:
{{
  "wakeword": true | false,
  "action": string | null,
  "params": object | null
}}

Rules:
- wakeword = true ONLY if the sentence starts with "Simon says", otherwise false.
- If no valid command is recognized, set action = null and params = null.
- Never invent actions or parameters.
- All numbers must be floats.
- If a parameter is not mentioned, omit it from params.
- params MUST be null if action is null.

supported actions and their parameters:

1. move_position
params:
{{
  "distance_m": number,
  "direction_vector": [number, number, number],
  "theta_deg": number
}}

Direction mapping:
forward = [1, 0, 0]
backward = [-1, 0, 0]
left = [0, -1, 0]
right = [0, 1, 0]
up = [0, 0, 1]
down = [0, 0, -1]

2. posture
params:
{{
  "posture_name": string
}}

3. move_joint
params:
{{
  "joint": string,
  "angle_deg": number
}}

Valid joint names:
{", ".join(self.joints)}

Valid joint angle ranges (degrees):
{", ".join([f'"{joint}": {rng}' for joint, rng in self.joints_angle_ranges.items()])}

4. change_eye_color
params:
{{
  "color": "red" | "green" | "blue" | "yellow"
}}

5. capture_frame
params:
{{}}

6. say_text
params:
{{
  "text": string
}}

Important:
- Only include parameters that are explicitly mentioned.
- Do NOT guess default values.
- If the user intent is unclear, return action = null.

Sentence: "{text}"
""".strip()
        self.model.eval()
        inputs = self.tokenizer(prompt, return_tensors="pt")
        inputs = {k: v.to(self.model.device) for k, v in inputs.items()}

        with torch.no_grad():
            out = self.model.generate(**inputs, max_new_tokens=160)

        gen = out[0, inputs["input_ids"].shape[1]:]
        raw = self.tokenizer.decode(gen, skip_special_tokens=True).strip()

        try:
            return json.loads(raw)
        except json.JSONDecodeError:
            pass

        matches = re.findall(r"\{.*?\}", raw, re.DOTALL)
        if matches:
            return json.loads(matches[-1])

        raise ValueError(f"Did not receive valid JSON response from model. Raw: {raw[:200]}...")

if __name__ == '__main__':
    import os
    parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
    model_dir = os.path.join(parent_dir, "models", "qwen")
    recorder = NaoVoiceCommand(model_dir)
    command = recorder.extract_command("Simon says look with red eyes.")
    print(command)