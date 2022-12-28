

import Leap, sys, thread, time
import pandas as pd
import time,csv
import os, threading

startTime = str(int(round(time.time())))

print startTime







class SampleListener(Leap.Listener):
    finger_names = ['Thumb', 'Index', 'Middle', 'Ring', 'Pinky']
    bone_names = ['Metacarpal', 'Proximal', 'Intermediate', 'Distal']
    state_names = ['STATE_INVALID', 'STATE_START', 'STATE_UPDATE', 'STATE_END']
    def on_init(self, controller):
        # imagesAllowed = controller.config.get("tracking_images_mode") ==
        controller.set_policy(Leap.Controller.POLICY_IMAGES)
    def on_connect(self, controller):
        print "Connected"
        global first_time
        first_time=time.time()

    def on_disconnect(self, controller):
        # Note: not dispatched when running in a debugger.
        print "Disconnected"
    def on_exit(self, controller):
        print "Exited"
    def on_frame(self, controller):
        # controller.set_policy(Leap.Controller.POLICY_OPTIMIZE_HMD)
        time.sleep(0.0001)
        # Get the most recent frame and report some basic information
        frame = controller.frame()

        for hand in frame.hands:

            # you may need to switch these around depending on your Leap Motion
            handType = "Left hand" if hand.is_left else "Right hand"

            timestamp = frame.timestamp
            if True:
                hand_pinch=0
                hand_strength=0

            hand_pinch = hand.pinch_strength
            hand_strength = hand.grab_strength
            extended_fingers=frame.fingers.extended()

        if not (frame.hands.is_empty and frame.gestures().is_empty):


            recorded_variables = [hand_pinch, hand_strength, hand_pinch+hand_strength]
            line = ""
            for attribute in recorded_variables:
                line = line + str(attribute) + ","
            #line = line + "/" + imageName
            line = line #+ "," + move
            print line
        # print line






def main():
    # Create a sample listener and controller
    listener = SampleListener()
    controller = Leap.Controller()

    # Have the sample listener receive events from the controller
    controller.add_listener(listener)

    # Keep this process running until Enter is pressed
    print "Press Enter to quit..."
    try:
        sys.stdin.readline()
    except KeyboardInterrupt:
        pass
    finally:
        # Remove the sample listener when done
        controller.remove_listener(listener)


if __name__ == "__main__":
    main()