import sys
import json

from src.tasks import NaoTaskExecutor

robot_ip = sys.argv[1]
port = int(sys.argv[2])
payload_json = sys.argv[3]
print(robot_ip, port, payload_json)

robot = NaoTaskExecutor(robot_ip, port)

