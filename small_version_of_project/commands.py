from interfaces import IStartSystem, IEndSystem, IStartTest, IEndTest, ICommands, IShowVideo, ITerminalManager
import cv2
import sys



class StartSystem(IStartSystem):
    @staticmethod
    def start_system(model):
        if model.system_status:
            return
        model.system_status = True
        with model.terminal_lock:
            print("System has started")

    
class EndSystem(IEndSystem):
    @staticmethod
    def end_system(model):
        model.system_status = False
        TerminalOutputManager.release_terminal(model)
        with model.terminal_lock:
            print("System has ended")

    
    @staticmethod
    def terminate(video, pre_process):
        if pre_process.camera_type == "picamera2": # if on linux
            video.stop()
            video.close()
        else: #If litterally anywhere else
            video.release()
        cv2.destroyAllWindows()
        sys.exit("Ended the system and released video and destroyed all windows and released all threads")
        


class StartTest(IStartTest):
    @staticmethod
    def start_test(model):
        if model.test_mode:
            return
        model.test_mode = True
        model.terminal_mode = "model"
        TerminalOutputManager.release_terminal(model)
        with model.terminal_lock:
            print("Starting Test")
   

class EndTest(IEndTest):
    @staticmethod
    def end_test(model):
        model.test_mode = False
        with model.terminal_lock:
            print("Ending test")


class ShowVideo(IShowVideo):
    @staticmethod
    def show_video(model):
        model.show_recording = True
        with model.terminal_lock:
            print("Will show recording")
    @staticmethod
    def end_video(model):
        model.show_recording = False
        with model.terminal_lock:
            print("Stop recording")


class TerminalOutputManager(ITerminalManager):
    @staticmethod
    def lock_terminal(model):
        if model.terminal_mode == "user":
            return
        model.terminal_mode = "user"
        ShowVideo.end_video(model)
        with model.terminal_lock:       
            print("Terminal is now open for user input")
    
    @staticmethod
    def release_terminal(model):
        if model.terminal_mode == "model":
            return
        model.terminal_mode = "model"
        with model.terminal_lock: 
            print("Terminal is now open to model input")
   
class Commands(ICommands):
    def __init__(self, model):
       self.model = model
       self.commands = {
            "ss": StartSystem.start_system,
            "es": EndSystem.end_system,
            "st": StartTest.start_test,
            "et": EndTest.end_test,
            "v": ShowVideo.show_video,
            "l": TerminalOutputManager.lock_terminal,
            "r": TerminalOutputManager.release_terminal

        }
       
    def execute(self, ui) -> bool:
        action = self.commands.get(ui)
        if ui == "es":
            EndSystem.terminate(self.model.video, self.model.pre_process_camera)
        if action:
            action(self.model)
            return True

        print(f"Unknown command: {ui}")
        return False

