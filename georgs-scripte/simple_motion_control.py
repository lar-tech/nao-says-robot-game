# -*- encoding: UTF-8 -*-
"""Example: Use angleInterpolation Method"""

import sys
import time
from naoqi import ALProxy

def main(robotIP):
    PORT = 9559

    try:
        motionProxy = ALProxy("ALMotion", robotIP, PORT)
    except Exception,e:
        print "Could not create proxy to ALMotion"
        print "Error was: ",e
        sys.exit(1)

    motionProxy.setStiffnesses("Head", 1.0)

    # Example showing multiple trajectories
    names      = ["HeadYaw", "HeadPitch"]
    angleLists = [[-1.0, -0.5, 0.0, 0.5, 1.0], [0.5, 0.25, 0.0, -0.25, -0.5]]
    timeLists  = [[1.0, 2.0, 3.0, 4.0, 5.0], [1.0, 2.0, 3.0, 4.0, 5.0]]
    isAbsolute = True
    motionProxy.angleInterpolation(names, angleLists, timeLists, isAbsolute)

    time.sleep(1.0)
    motionProxy.setStiffnesses("Head", 0.0)

if __name__ == "__main__":
    robotIp = "127.0.0.1"

    if len(sys.argv) <= 1:
        print "Usage python motion_angleInterpolation.py robotIP (optional default: 127.0.0.1)"
    else:
        robotIp = sys.argv[1]

    main(robotIp)