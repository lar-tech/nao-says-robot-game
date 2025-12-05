# -*- encoding: UTF-8 -*-
"""Enhanced Example: Use angleInterpolation Method with Audio Feedback"""

import sys
import time
from naoqi import ALProxy

def main(robotIP):
    PORT = 9559

    try:
        motionProxy = ALProxy("ALMotion", robotIP, PORT)
        ttsProxy = ALProxy("ALTextToSpeech", robotIP, PORT)  # Added for feedback
    except Exception,e:
        print "Could not create proxy to ALMotion or ALTextToSpeech"
        print "Error was: ",e
        sys.exit(1)

    # Make the robot introduce the movement
    ttsProxy.say("Watch me move my head smoothly!")
    
    motionProxy.setStiffnesses("Head", 1.0)

    # Enhanced: Add a waving motion after the head movement
    names      = ["HeadYaw", "HeadPitch"]
    angleLists = [[-1.0, -0.5, 0.0, 0.5, 1.0], [0.5, 0.25, 0.0, -0.25, -0.5]]
    timeLists  = [[1.0, 2.0, 3.0, 4.0, 5.0], [1.0, 2.0, 3.0, 4.0, 5.0]]
    isAbsolute = True
    motionProxy.angleInterpolation(names, angleLists, timeLists, isAbsolute)

    # Enhanced: Add a fun waving gesture
    ttsProxy.say("Now let me wave hello!")
    motionProxy.angleInterpolation("RShoulderPitch", [-1.0, -0.5, -1.0, -0.5, -1.0], [1.0, 1.5, 2.0, 2.5, 3.0], True)

    time.sleep(1.0)
    motionProxy.setStiffnesses("Head", 0.0)
    ttsProxy.say("Movement complete!")

if __name__ == "__main__":
    robotIp = "127.0.0.1"

    if len(sys.argv) <= 1:
        print "Usage python motion_angleInterpolation.py robotIP (optional default: 127.0.0.1)"
    else:
        robotIp = sys.argv[1]

    main(robotIp)