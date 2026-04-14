from interfaces import IStartSystem, IEndSystem, IStartTest, IEndTest
import cv2
import sys


class StartSystem(IStartSystem):
    @staticmethod
    def start_system(model):
        model.system_status = True
        print("System has started")

    
class EndSystem(IEndSystem):
    @staticmethod
    def end_system(model):
        model.system_status = False
        print("System has ended")

    @staticmethod
    def terminate(video, pre_process):
        
        if pre_process.camera_type == "picamera2": # if on linux
            video.stop()
            video.close()
        else: #If litterally anywhere else
            video.release()
        cv2.destroyAllWindows()
        sys.exit("Ended the system and released video and destroyed all windows")


class StartTest(IStartTest):
    @staticmethod
    def start_test(model):
        model.test_mode = True
        print("Starting Test")
   

class EndTest(IEndTest):
    @staticmethod
    def end_test(model):
        model.test_mode = False
        print("Ending test")
   