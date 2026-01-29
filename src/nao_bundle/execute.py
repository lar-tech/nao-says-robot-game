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
        dist = params.get("distance_m", 0.0)
        vec = params.get("direction_vector", [0.0, 0.0, 0.0])
        theta_deg = params.get("theta_deg", 0.0)
        x = float(vec[0]) * float(dist)
        y = float(vec[1]) * float(dist)
        theta = to_rad(theta_deg)
        return executor.move_position(x=x, y=y, theta=theta)

    if action == "posture":
        posture_name = to_str(params.get("posture_name", "StandInit"))
        return executor.posture(posture_name=posture_name, speed=1.0)

    if action == "move_joint":
        joint = to_str(params.get("joint", "HeadYaw"))
        angle = to_rad(params.get("angle_deg", 0.0))
        return executor.move_joint(joint_name=joint, angle=angle, speed=0.1, waitingtime=2.0)

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
    sys.stderr.write("Received command: {}\n".format(cmd))
    executor = NaoTaskExecutor(ip, port)

    try:
        dispatch(executor, cmd)
    finally:
        executor.close()

if __name__ == "__main__":
    main()