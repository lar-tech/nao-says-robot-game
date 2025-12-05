# -*- encoding: UTF-8 -*-
"""Enhanced Example: Use Basic Awareness with Status Logging"""

import sys
import time
from naoqi import ALProxy

def main(robotIP):
    PORT = 9559

    try:
        basic_awareness = ALProxy("ALBasicAwareness", robotIP, PORT)
        ttsProxy = ALProxy("ALTextToSpeech", robotIP, PORT)  # Added for feedback
        memoryProxy = ALProxy("ALMemory", robotIP, PORT)     # Added for status monitoring
    except Exception,e:
        print "Could not create proxy to ALBasicAwareness, ALTextToSpeech, or ALMemory"
        print "Error was: ",e
        sys.exit(1)

    # Enhanced: Introduction
    ttsProxy.say("Activating basic awareness. I'll let you know what I notice.")

    # start
    basic_awareness.startAwareness()

    print "Basic Awareness started."
    print "Use Ctrl+c to stop this script."

    lastStatus = ""
    try:
        while True:
            # Enhanced: Monitor awareness status and human detection
            status = memoryProxy.getData("ALBasicAwareness/StimulusDetected")
            humanPresence = memoryProxy.getData("ALBasicAwareness/HumanPresenceDetected")
            
            if status != lastStatus:
                if status:
                    print "Stimulus detected:", status
                    ttsProxy.say("I noticed something interesting!")
                lastStatus = status
            
            if humanPresence:
                print "Human presence detected"
                # Only say this once per detection period
                if not hasattr(main, 'humanGreeted'):
                    ttsProxy.say("I sense someone nearby!")
                    main.humanGreeted = True
            else:
                if hasattr(main, 'humanGreeted'):
                    delattr(main, 'humanGreeted')
            
            time.sleep(2)  # Check every 2 seconds
    except KeyboardInterrupt:
        print
        print "Interrupted by user"
        print "Stopping..."

    # stop
    basic_awareness.stopAwareness()
    ttsProxy.say("Basic awareness deactivated.")
    print "Basic Awareness stopped."

if __name__ == "__main__":
    robotIp = "127.0.0.1"

    if len(sys.argv) <= 1:
        print "Usage python albasicawareness_example.py robotIP (optional default: 127.0.0.1)"
    else:
        robotIp = sys.argv[1]

    main(robotIp)