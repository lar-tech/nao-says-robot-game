import os
import time
import sys
sys.path.insert(0, "/workspace/inao")
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
SRC_DIR = os.path.join(CURRENT_DIR, "..", "src")
sys.path.insert(0, SRC_DIR)
from inao import NAO
from naoqi import ALProxy

class SimonSaysGame:
    def __init__(self, ip, port):
        self.tts = ALProxy("ALTextToSpeech", ip, port)
        self.leds = ALProxy("ALLeds", ip, port)
        self.audio_recorder = ALProxy("ALAudioRecorder", ip, port)
        
    def change_eye_color(self, color):
        color_map = {
            "red": (1.0, 0.0, 0.0),
            "green": (0.0, 1.0, 0.0),
            "blue": (0.0, 0.0, 1.0),
            "yellow": (1.0, 1.0, 0.0)
            }
        if color in color_map:
            rgb = color_map[color]
            self.leds.fadeRGB("FaceLeds", rgb[0], rgb[1], rgb[2], 0.5)
        time.sleep(1)
        self.leds.fadeRGB("FaceLeds", 1.0, 1.0, 1.0, 0.3)
        time.sleep(0.3)

    def record_audio(self, duration):
        path_to_audio = "home/nao/recording.wav"
        sample_rate = 16e3
        channels = [1, 1, 1, 1]
        self.audio_recorder.startMicrophonesRecording(path_to_audio, "wav", sample_rate, channels)
        time.sleep(duration)
        self.audio_recorder.stopMicrophonesRecording()

        return path_to_audio

if __name__ == "__main__":
    ROBOT_IP = "192.168.1.118"
    PORT = 9559
    game = SimonSaysGame(ROBOT_IP, PORT)
    game.record_audio(5)
   