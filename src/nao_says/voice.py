import os
import json
import re
import threading

os.environ["HF_HUB_OFFLINE"] = "1"
from RealtimeSTT import AudioToTextRecorder

class NaoVoiceCommand():
    def __init__(self):
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
        print(1)
        self.recorder = AudioToTextRecorder(language="en", device="cpu", compute_type="float32")
        print(2)
        self.last_cmd = None
        self._done = threading.Event()
        print(3)

    def close(self):
        if self.recorder:
            self.recorder.shutdown()
            self.recorder = None

    def record_audio(self):
        self._done.clear()
        self.recorder.start()
        input("Press Enter to stop recording...")
        self.recorder.stop()
        self.recorder.text(self.process_audio)
        self._done.wait()
        
        return json.dumps(self.last_cmd, ensure_ascii=True)

    def process_audio(self, text):
        print("Recognized Text:", text)
        self.last_cmd = self.map_command(text)
        self._done.set()

    def map_command(self, text: str) -> dict:
        text_lower = text.lower().strip()
        
        # wakeword check
        wakeword = text_lower.startswith("simon says")
        if wakeword:
            text_lower = text_lower[len("simon says"):].strip()
        result = {"wakeword": wakeword, "action": None, "params": None}
        
        # colors for eye color
        colors = {"red", "green", "blue", "yellow"}
        
        # directions for movement
        directions = {
            "forward": [1, 0, 0], "backward": [-1, 0, 0],
            "left": [0, -1, 0], "right": [0, 1, 0],
            "up": [0, 0, 1], "down": [0, 0, -1]
        }
        
        # extract number
        def find_number(s):
            m = re.search(r"(\d+\.?\d*)", s)
            return float(m.group(1)) if m else None

        # action matching
        # change_eye_color
        if "eye" in text_lower and "color" in text_lower:
            for color in colors:
                if color in text_lower:
                    result["action"] = "change_eye_color"
                    result["params"] = {"color": color}
                    return result
        
        # capture_frame
        if any(kw in text_lower for kw in ["capture", "photo", "picture", "snapshot"]):
            result["action"] = "capture_frame"
            result["params"] = {}
            return result
        
        # say_text
        if text_lower.startswith("say "):
            spoken = text[len("simon says say "):] if wakeword else text[len("say "):]
            result["action"] = "say_text"
            result["params"] = {"text": spoken.strip()}
            return result
        
        # posture
        postures = ["stand", "sit", "crouch", "lyingbelly", "lyingback"]
        for posture in postures:
            if posture in text_lower:
                result["action"] = "posture"
                result["params"] = {"posture_name": posture.capitalize()}
                return result
        
        # move_position
        for direction, vec in directions.items():
            if direction in text_lower:
                params = {"direction_vector": vec}
                num = find_number(text_lower)
                if num:
                    if "degree" in text_lower or "°" in text_lower:
                        params["theta_deg"] = num
                    else:
                        params["distance_m"] = num
                result["action"] = "move_position"
                result["params"] = params
                return result
        
        # TODO: mapping raise left arm to predefined joint angles
        # e.g., raise left arm
        # move_joint
        for joint in self.joints:
            if joint.lower() in text_lower:
                angle = find_number(text_lower)
                if angle:
                    result["action"] = "move_joint"
                    result["params"] = {"joint": joint, "angle_deg": angle}
                    return result
        
        return result

if __name__ == '__main__':
    import os
    parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
    recorder = NaoVoiceCommand()
    command = recorder.record_audio()
    # command = recorder.map_command("Simon says sit down.")
    print(command)
    recorder.close()
