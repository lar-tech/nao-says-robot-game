import json
import re
import sys
import math

import numpy as np
from faster_whisper import WhisperModel
import pyaudio

class NaoVoiceCommand():
    def __init__(self, model_dir="./setup/whisper"):
        # joint settings
        self.commands = {
            # Postures
            "StandZero": {"function": "posture", "params": {"posture_name": "StandZero", "speed": 1.0}},
            "StandInit": {"function": "posture", "params": {"posture_name": "StandInit", "speed": 1.0}},
            "Stand": {"function": "posture", "params": {"posture_name": "Stand", "speed": 1.0}},
            "Crouch": {"function": "posture", "params": {"posture_name": "Crouch", "speed": 1.0}},
            "Sit": {"function": "posture", "params": {"posture_name": "Sit", "speed": 1.0}},
            "SitRelax": {"function": "posture", "params": {"posture_name": "SitRelax", "speed": 1.0}},
            "LyingBelly": {"function": "posture", "params": {"posture_name": "LyingBelly", "speed": 1.0}},
            "LyingBack": {"function": "posture", "params": {"posture_name": "LyingBack", "speed": 1.0}},
            # Movement
            "MoveForward": {"function": "move_position", "params": {"x": 0.2, "y": 0.0, "theta": 0.0}},
            "MoveBackward": {"function": "move_position", "params": {"x": -0.2, "y": 0.0, "theta": 0.0}},
            "MoveRight": {"function": "move_position", "params": {"x": 0.0, "y": -0.2, "theta": 0.0}},
            "MoveLeft": {"function": "move_position", "params": {"x": 0.0, "y": 0.2, "theta": 0.0}},
            "TurnRight": {"function": "move_position", "params": {"x": 0.0, "y": 0.0, "theta": -math.pi / 4}},
            "TurnLeft": {"function": "move_position", "params": {"x": 0.0, "y": 0.0, "theta": math.pi / 4}},
            # Head
            "RotateHead": {"function": "move_joint", "params": {"joint_name": "HeadYaw", "angle": math.pi / 4, "speed": 0.1, "waitingtime": 2.0}},
            "MoveHead": {"function": "move_joint", "params": {"joint_name": "HeadPitch", "angle": -10, "speed": 0.1, "waitingtime": 2.0}},
            # Arms
            "LiftLeftArmFront": {"function": "move_joint", "params": {"joint_name": "LShoulderPitch", "angle": 0, "speed": 0.1, "waitingtime": 2.0}},
            "LiftRightArmFront": {"function": "move_joint", "params": {"joint_name": "RShoulderPitch", "angle": 0, "speed": 0.1, "waitingtime": 2.0}},
            "LiftLeftArmSide": {"function": "move_joint", "params": {"joint_name": "LShoulderRoll", "angle": 60, "speed": 0.1, "waitingtime": 2.0}},
            "LiftRightArmSide": {"function": "move_joint", "params": {"joint_name": "RShoulderRoll", "angle": -60, "speed": 0.1, "waitingtime": 2.0}},
            "StretchLeftElbow": {"function": "move_joint", "params": {"joint_name": "LElbowRoll", "angle": -2, "speed": 0.1, "waitingtime": 2.0}},
            "BendLeftElbow": {"function": "move_joint", "params": {"joint_name": "LElbowRoll", "angle": -88, "speed": 0.1, "waitingtime": 2.0}},
            "StretchRightElbow": {"function": "move_joint", "params": {"joint_name": "RElbowRoll", "angle": 2, "speed": 0.1, "waitingtime": 2.0}},
            "BendRightElbow": {"function": "move_joint", "params": {"joint_name": "RElbowRoll", "angle": 88, "speed": 0.1, "waitingtime": 2.0}},
            "TwistLeftWrist": {"function": "move_joint", "params": {"joint_name": "LWristYaw", "angle": -90, "speed": 0.1, "waitingtime": 2.0}},
            "TwistRightWrist": {"function": "move_joint", "params": {"joint_name": "RWristYaw", "angle": 90, "speed": 0.1, "waitingtime": 2.0}},}
        
        self.voice_map = {
            # Postures
            "stand zero": "StandZero",
            "stand init": "StandInit",
            "stand": "Stand",
            "crouch": "Crouch",
            "sit down": "Sit",
            "sit": "Sit",
            "sit relax": "SitRelax",
            "relax": "SitRelax",
            "lie down": "LyingBelly",
            "lying belly": "LyingBelly",
            "lying back": "LyingBack",
            "lie on back": "LyingBack",
            # Movement
            "move forward": "MoveForward",
            "go forward": "MoveForward",
            "forward": "MoveForward",
            "move backward": "MoveBackward",
            "go backward": "MoveBackward",
            "backward": "MoveBackward",
            "move right": "MoveRight",
            "go right": "MoveRight",
            "move left": "MoveLeft",
            "go left": "MoveLeft",
            "turn right": "TurnRight",
            "turn left": "TurnLeft",
            # Head
            "rotate head": "RotateHead",
            "turn head": "RotateHead",
            "move head": "MoveHead",
            "look down": "MoveHead",
            # Arms
            "lift left arm": "LiftLeftArmFront",
            "raise left arm": "LiftLeftArmFront",
            "lift right arm": "LiftRightArmFront",
            "raise right arm": "LiftRightArmFront",
            "left arm side": "LiftLeftArmSide",
            "right arm side": "LiftRightArmSide",
            "stretch left elbow": "StretchLeftElbow",
            "bend left elbow": "BendLeftElbow",
            "stretch right elbow": "StretchRightElbow",
            "bend right elbow": "BendRightElbow",
            "twist left wrist": "TwistLeftWrist",
            "twist right wrist": "TwistRightWrist",}

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

        # action matching
        # change_eye_color
        colors = {"red", "green", "blue", "yellow"}
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
        
        # move_position, posture, move_joint
        for phrase in sorted(self.voice_map.keys(), key=len, reverse=True):
            if phrase in text_lower:
                cmd_key = self.voice_map[phrase]
                cmd = self.commands[cmd_key]
                result["action"] = cmd["function"]
                result["params"] = cmd["params"]
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
    command = recorder.get_command()
    print(command)

