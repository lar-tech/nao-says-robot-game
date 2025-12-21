import json
import math
import sys

from tasks import NaoTaskExecutor

def to_rad(deg):
    return float(deg) * math.pi / 180.0

def dispatch(executor, cmd):
    if not cmd.get("wakeword", False):
        executor.tts_proxy.say("No wakeword.")
        return

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
        return executor.posture(posture_name=params.get("posture_name", "StandInit"), speed=1.0)

    if action == "move_joint":
        joint = params.get("joint", "HeadYaw")
        angle = to_rad(params.get("angle_deg", 0.0))
        return executor.move_joint(joint_name=joint, angle=angle, speed=0.1, waitingtime=2.0)

    if action == "change_eye_color":
        return executor.change_eye_color(params.get("color", "yellow"))

    if action == "capture_frame":
        img_b64 = executor.capture_frame()
        sys.stdout.write(img_b64)
        return

    if action == "say_text":
        return executor.tts_proxy.say(params.get("text", ""))

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