from feed_data import Camera, PreProcessCamera
import threading
import queue
from interfaces import ICommands
from commands import StartSystem, EndSystem, StartTest, EndTest
import time
import cv2



class CallModel:
    def __init__(self):
        self.test_mode = False
        self.system_status = False
        self.running = True
        

        self.command_queue = queue.Queue()
        self.command_handler = Commands(self)
        self.output_queue = queue.Queue()
        self.pre_process_camera = PreProcessCamera()
        self.video = self.pre_process_camera.open_camera()
        
    def input_listener(self):
        while self.running:
            ui = input("Enter command: ").strip()
            self.command_queue.put(ui)

    def process_commands(self):
        while not self.command_queue.empty():
            cmd = self.command_queue.get()
            self.command_handler.execute(cmd)
    
    def process_output(self):
        while not self.output_queue.empty():
            print(self.output_queue.get())
            

    def testing_model(self):
        input_thread = threading.Thread(target=self.input_listener, daemon=True)
        input_thread.start()
        dropout_prob=0.2
        camera = Camera(dropout_prob, self)

       
        
        while self.running:
            self.process_commands()
            self.process_output()
           
            if not self.system_status:
                print("System is off. Enter 'ss' to start the system.")
                time.sleep(0.5)
                
            camera.get_video(self.output_queue)
            time.sleep(0.01)
            
            
        
        
class Commands(ICommands):
    def __init__(self, model):
       self.model = model
       self.commands = {
            "ss": StartSystem.start_system,
            "es": EndSystem.end_system,
            "st": StartTest.start_test,
            "et": EndTest.end_test
        }
       
    def execute(self, ui) -> bool:
        action = self.commands.get(ui)
        if ui == "es":
            EndSystem.terminate(self.model.video)
        if action:
            action(self.model)
            return True

        print(f"Unknown command: {ui}")
        return False

call_model = CallModel()
call_model.testing_model()



