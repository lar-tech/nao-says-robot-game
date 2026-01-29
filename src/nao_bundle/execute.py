import json
import math
import sys

from tasks import NaoTaskExecutor

def to_rad(deg):
    return float(deg) * math.pi / 180.0

def to_str(s):
    """Convert unicode to str for Python 2.7 compatibility."""
    try:
        if isinstance(s, unicode):
            return s.encode('utf-8')
    except NameError:
        # Python 3: unicode doesn't exist
        pass
    return s

def dispatch(executor, cmd):
    action = cmd.get("action")
    params = cmd.get("params") or {}

    if action == "move_position":
        x = float(params.get("x", 0.0))
        y = float(params.get("y", 0.0))
        theta = float(params.get("theta", 0.0))  # already in radians from voice.py
        return executor.move_position(x=x, y=y, theta=theta)

    if action == "posture":
        posture_name = to_str(params.get("posture_name", "StandInit"))
        return executor.posture(posture_name=posture_name, speed=1.0)

    if action == "move_joint":
        joint = to_str(params.get("joint_name", "HeadYaw"))
        angle = float(params.get("angle", 0.0))  # already in radians from voice.py
        speed = float(params.get("speed", 0.1))
        waitingtime = float(params.get("waitingtime", 2.0))
        return executor.move_joint(joint_name=joint, angle=angle, speed=speed, waitingtime=waitingtime)

    if action == "change_eye_color":
        color = to_str(params.get("color", "yellow"))
        return executor.change_eye_color(color)

    if action == "capture_frame":
        img_b64 = executor.capture_frame()
        sys.stdout.write(img_b64)
        return

    if action == "say_text":
        text = to_str(params.get("text", ""))
        return executor.tts_proxy.say(text)

    if action == "game_over":
        executor.tts_proxy.say("It was pleasent to meet you.")
        return

    executor.tts_proxy.say("Unknown action.")
    return

def main():
    ip = sys.argv[1]
    port = int(sys.argv[2])
    cmd = json.loads(sys.stdin.read() or "{}")
    executor = NaoTaskExecutor(ip, port)

    try:
        dispatch(executor, cmd)
    finally:
        executor.close()

if __name__ == "__main__":
    main()