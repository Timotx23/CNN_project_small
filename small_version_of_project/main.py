from feed_data import Camera, PreProcessCamera
import threading
import queue
from commands import Commands
import time




class CallModel:
    def __init__(self, show_video):
        self.test_mode = False
        self.system_status = False
        self.running = True
        self.show_video = show_video

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
                
            camera.get_video(self.output_queue, self.show_video)
            time.sleep(0.01)
            
            
        
        
show_vid = False
call_model = CallModel(show_vid)
call_model.testing_model()


#TODO -> Show video could be made better as an actuall terminal command
#TODO -> Clean up main.py seperate the threading cuz its messy and doesnt have to be there
#TODO -> Terminal input is still messi ie it doesn't let me type anything cleanly without it being burried making it confusing if I have alr printed something or not