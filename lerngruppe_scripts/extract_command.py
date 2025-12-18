import json
import re
import requests
import sys
import numpy as np
from transformers import AutoProcessor, AutoModelForSpeechSeq2Seq
import torch
import soundfile as sf
import sounddevice as sd



OLLAMA_URL = "http://localhost:11434/api/generate"
MODEL_NAME = "qwen2.5:1.5b-instruct"

ALLOWED_KEYS = {"wakeword", "action", "posture_name", "distance_m", "direction_vector", "joint", "angle_deg"}



JOINTS = [
    "HeadYaw","HeadPitch",
    "LShoulderPitch","LShoulderRoll","LElbowYaw","LElbowRoll","LWristYaw","LHand",
    "RShoulderPitch","RShoulderRoll","RElbowYaw","RElbowRoll","RWristYaw","RHand",
    "HipYawPitch","LHipRoll","LHipPitch","LKneePitch","LAnklePitch","LAnkleRoll",
    "RHipRoll","RHipPitch","RKneePitch","RAnklePitch","RAnkleRoll"
]
JOINTS_ANGLE_RANGES = {
    "HeadYaw": (-119.5, 119.5), "HeadPitch": (-38.5, 29.5),
    "LShoulderPitch": (-119.5, 119.5), "LShoulderRoll": (-18.0, 76.0), "LElbowYaw": (-119.5, 119.5), "LElbowRoll": (-88.5, -2.0), "LWristYaw": (-104.5, 104.5), "LHand": (0.0, 1.0),
    "RShoulderPitch": (-119.5, 119.5), "RShoulderRoll": (-76.0, 18.0), "RElbowYaw": (-119.5, 119.5), "RElbowRoll": (2.0, 88.5), "RWristYaw": (-104.5, 104.5), "RHand": (0.0, 1.0),
    "HipYawPitch": (-65.62, 42.48), "LHipRoll": (-21.73, 45.0), "LHipPitch": (-88.0, 27.73), "LKneePitch": (0.0, 121.0), "LAnklePitch": (-68.0, 52.87), "LAnkleRoll": (-22.84, 44.5),
    "RHipRoll": (-45.0, 21.73), "RHipPitch": (-88.0, 27.73), "RKneePitch": (0.0, 121.0), "RAnklePitch": (-68.0, 52.87), "RAnkleRoll": (-44.5, 22.84),
}

def _extract_json(text: str) -> dict:
    """
    Versucht, ein JSON-Objekt aus der Modellantwort zu extrahieren.
    Funktioniert auch, wenn das Modell doch etwas Text davor/danach schreibt.
    """
    text = text.strip()

    # 1) Direktes JSON?
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass

    # 2) Erstes {...} suchen
    m = re.search(r"\{.*\}", text, re.DOTALL)
    if m:
        return json.loads(m.group(0))

    raise ValueError("Keine gültige JSON-Antwort vom Modell erhalten.")

def extract_command(text: str) -> dict:
    prompt = f"""
Du bist ein Parser für NAO-Roboterkommandos.
Gib NUR ein JSON-Objekt aus. Kein Markdown. Keine zusätzlichen Schlüssel.

Erlaubte Schlüssel (genau diese, keine weiteren):
wakeword, action, distance_m, direction_vector, joint, angle_deg

Schema:
{{
  "wakeword": true/false,
  "action": "move_position"|"posture"|"move_joint",
  "distance_m": number|null,
  "direction_vector": [number,number,number]|null,
  "joint": string|null,
  "angle_deg": number|null
}}

Regeln:
- wakeword = true, wenn der Satz mit "NAO says" beginnt, sonst false.
- Zahlen immer als float (z.B. 1.0, 30.0).
- Wenn etwas nicht vorhanden ist: null (nicht false).
- wenn du nichts erkennen kannst, gib alle Werte als null/false zurück.
- action muss eine der vorgegebenen Aktionen sein, wenn du sicher bist das überhaupt eine dabei ist.

Mapping direction_vector:
forward=[1,0,0], backward=[-1,0,0], left=[0,-1,0], right=[0,1,0], up=[0,0,1], down=[0,0,-1]

Gelenknamen (joint muss exakt einer davon sein, wenn einer erkannt wird):
{", ".join(JOINTS)}
Gelenkwinkel (angle_deg) immer in Grad. Je nach Gelenk verschiedene Bereiche beachten. Nur wenn auch ein Gelenk erkannt wurde.
Bereiche der Gelenkwinkel (angle_deg):
{", ".join([f'"{joint}": {rng}' for joint, rng in JOINTS_ANGLE_RANGES.items()])}

Satz: "{text}"
""".strip()

    r = requests.post(
        OLLAMA_URL,
        json={
            "model": MODEL_NAME,
            "prompt": prompt,
            "stream": False,
            "options": {"temperature": 0.0, "num_predict": 160},
        },
        timeout=60,
    )
    r.raise_for_status()
    raw = r.json()["response"]

    return _extract_json(raw)

def record_wav(filename="output.wav", seconds=5, rate=16000, channels=1):
    print("* recording")
    audio = sd.rec(int(seconds * rate), samplerate=rate, channels=channels, dtype="float32")
    sd.wait()
    print("* done recording")

    # soundfile erwartet (samples, channels)
    sf.write(filename, audio, rate)
    return filename

def speech_to_text(filename: str) -> str:
    # load model
    model_path = "./models"   
    processor = AutoProcessor.from_pretrained(model_path)
    model = AutoModelForSpeechSeq2Seq.from_pretrained(model_path,dtype=torch.float32).to("cuda" if torch.cuda.is_available() else "cpu")
    model.eval()

    # read audio file
    audio, sr = sf.read(filename)
    inputs = processor(audio, sampling_rate=sr, return_tensors="pt").to(model.device)
    with torch.no_grad():
        ids = model.generate(**inputs)

    text = processor.batch_decode(ids, skip_special_tokens=True)[0]
    return text

if __name__ == "__main__":
    #recorded_filename = record_wav(filename="lerngruppe_scripts/recorded_command.wav")
    audio_file = sys.argv[1] 
    spoken_text = speech_to_text(audio_file)
    result = extract_command(spoken_text)
    print(json.dumps(result))












