import sys
import base64

from src.tasks import NaoTaskExecutor

# get args
ip = sys.argv[1]
port = int(sys.argv[2])
command_json = sys.argv[3]

executor = NaoTaskExecutor(ip, port)

# test commands
sys.stderr.write("Test")
jpeg = executor.capture_frame()
if jpeg is None:
    sys.stderr.write("No frame\n")
    sys.exit(1)
b64 = base64.b64encode(jpeg)
sys.stdout.write(b64)