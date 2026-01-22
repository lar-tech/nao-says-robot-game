import json
import re
import sys

import numpy as np
from faster_whisper import WhisperModel
import pyaudio

class NaoVoiceCommand():
    def __init__(self, model_dir="./setup/whisper"):
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
        self.model = WhisperModel("small", device="cpu", compute_type="float32", download_root=model_dir, local_files_only=True)
        self.audio = pyaudio.PyAudio()
        self.sample_rate = 16000
        self.wake_word = "simon says"

    def record_audio(self, duration=5, show_progress=True):
        stream = self.audio.open(format=pyaudio.paInt16, channels=1, rate=self.sample_rate, input=True, frames_per_buffer=1024)
        frames = []
        total_chunks = int(self.sample_rate / 1024 * duration)
        for i in range(total_chunks):
            data = stream.read(1024)
            frames.append(np.frombuffer(data, dtype=np.int16))

            if show_progress:
                progress = (i + 1) / total_chunks
                bar_len = 30
                filled = int(bar_len * progress)
                bar = "█" * filled + "░" * (bar_len - filled)
                remaining = duration * (1 - progress)
                sys.stdout.write(f"\r [{bar}] {remaining:.1f}s ")
                sys.stdout.flush()
        
        if show_progress:
            print()

        stream.stop_stream()
        stream.close()
        audio_data = np.concatenate(frames).astype(np.float32) / 32768.0
        return audio_data
    
    def transcribe(self, audio_data):
        segments, _ = self.model.transcribe(audio_data, language="en")
        return " ".join(segment.text for segment in segments).strip()
    
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

    def get_command(self):
        while True:
            print("Listening for 'Simon says'...")
            audio = self.record_audio(duration=5, show_progress=True)
            transcript = self.transcribe(audio).lower()

            if self.wake_word in transcript:
                # get command after wake word
                if len(transcript) > len(self.wake_word) + 3:
                    result = self.map_command(transcript)
                    result["wakeword"] = True
                    return json.dumps(result)
                
                # listen for command
                print("Wake word detected. Listening for command...")
                audio = self.record_audio(duration=5, show_progress=True)
                transcript = self.transcribe(audio).lower()                
                result = self.map_command(transcript)
                result["wakeword"] = True
                return json.dumps(result)

if __name__ == '__main__':
    recorder = NaoVoiceCommand()
    command = recorder.get_command(duration=5)
    print(command)
