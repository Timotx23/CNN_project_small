from interfaces import ISimController
import queue
import threading
from commands import Commands
from feed_data import CameraPreper, LoadModel, TensorizedFrame, Camera
import time
import model
import torch

class UserInputThreading:
    def __init__(self, owner):
        """This class controlls all threading that happens in the simulation system"""
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

class SimController(ISimController):
    def __init__(self):
        """This class acts as the complete controller of the simulation system it ensures that everything that needs to fed to all classes gets fed etc
        It is also an intermediate step from the actuall calling of my model and the execution
        """
 
        self._prep_threading()
        self._prep_system()
        self._prep_model()
        
    
    def _prep_threading(self):
        """ This function sets up all the things needed for the threading"""
        self.test_mode = False
        self.system_status = False
        self.running = True
        self.show_recording = False
        self.terminal_lock = threading.Lock()
        self.terminal_mode = "user"
        self.ready_for_input = threading.Event()
        self.ready_for_input.set()
        self.input_queue = UserInputThreading(self)
    
    def _prep_system(self):
        """ This function sets up the actual system itself in order for the simulation system"""
        self.command_handler = Commands(self)
        self.pre_process_camera = CameraPreper()
        self.video = self.pre_process_camera.open_camera()
    
    def _prep_model(self):
        """ These are the hyper parameters for the CNN model itself """
        self.dropout_prob=0.2
        self.height = 32
        self.width = 32
        self.rgb = 3
        self.device = model.CNN_model.to_devices()
        self.trained_weights = torch.load("model/model_d2.pth", map_location=self.device)
                
    def camera_preprocessing(self):
        if self.pre_process_camera.get_camera_path() == False:
            raise ValueError("No usable camera could be found.")
        if self.pre_process_camera.open_camera() == False:
            raise ValueError("Unknown camera type.")
        return True
    
    def call_model(self):
        input_thread = threading.Thread(target=self.input_queue.input_listener, daemon=True)
        input_thread.start()
        camera = Camera(self)
        return camera
    
    

    def load_model(self):
        load_model: LoadModel = LoadModel(self.dropout_prob, self.device, self.trained_weights)
        return load_model
    
    def system_setup(self):
        if self.camera_preprocessing() == True and self.load_model() != False:
            return True
        

    def load_tensorized_frame(self):
        tensorizedframe: TensorizedFrame = TensorizedFrame(self.height, self.width, self.rgb, self.device)
        return tensorizedframe

    def running_model(self,camera):
        self.input_queue.process_commands()
        self.input_queue.process_output()
        camera.get_video(self.input_queue.output_queue, self.show_recording)
        time.sleep(0.01)


    
    
    
            
    
    