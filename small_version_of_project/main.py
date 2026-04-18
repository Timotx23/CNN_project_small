from feed_data import Camera, PreProcessCamera
import threading
import queue
from commands import Commands
import time

class UserInputQueue:
    def __init__(self, owner):
        self.owner = owner
        self.command_queue = queue.Queue()
        
        self.output_queue = queue.Queue()
    def input_listener(self):
        if self.owner.system_status == False:
            with self.owner.terminal_lock: 
                print("System is off. Enter 'ss' to start the system.")
            
                
        while self.owner.running:
            self.owner.ready_for_input.wait()
            ui = input("Enter command: ").strip()
            if ui:
                self.command_queue.put(ui)

    def process_commands(self):
        while not self.command_queue.empty():
                cmd = self.command_queue.get()
                self.owner.command_handler.execute(cmd)
                self.owner.ready_for_input.set()
    
    def process_output(self):

        while not self.output_queue.empty():
            if not self.command_queue.empty():
                break  # prioritize commands
            msg = self.output_queue.get()
            with self.owner.terminal_lock:
                print(msg)



class CallModel:
    def __init__(self):
        self.test_mode = False
        self.system_status = False
        self.running = True
        self.show_recording = False
        self.terminal_lock = threading.Lock()
        self.terminal_mode = "user"
        self.command_handler = Commands(self)
        self.pre_process_camera = PreProcessCamera()
        self.ready_for_input = threading.Event()
        self.ready_for_input.set()
        self.input_queue = UserInputQueue(self)
        self.video = self.pre_process_camera.open_camera()
        
        
    
    def run_model(self):
        input_thread = threading.Thread(target=self.input_queue.input_listener, daemon=True)
        input_thread.start()
        dropout_prob=0.2
        camera = Camera(dropout_prob, self)
 
        while self.running:
            self.input_queue.process_commands()
            self.input_queue.process_output()
            camera.get_video(self.input_queue.output_queue, self.show_recording)
            time.sleep(0.01)
            

call_model = CallModel()
call_model.run_model()


#TODO -> update the UML Diagram to the needed standard
#TODO -> Figure out a way to make model more efficent

