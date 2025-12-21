import sys
import base64

from tasks import NaoTaskExecutor

# # get args
# ip = sys.argv[1]
# port = int(sys.argv[2])
# command_json = sys.argv[3]

executor = NaoTaskExecutor("192.168.1.118", 9559)

# # test commands
# sys.stderr.write("Test")
# sys.stdout.write("Test")
# jpeg = executor.capture_frame()
# if jpeg is None:
#     sys.stderr.write("No frame\n")
#     sys.exit(1)
# b64 = base64.b64encode(jpeg)
# sys.stdout.write(b64)
