import cv2
import platform
from cv2_enumerate_cameras import enumerate_cameras
import torch
import numpy as np
import model.CNN_model 

HEIGHT = 32
WIDTH = 32
RGB = 3
device = model.CNN_model.to_devices()
trained_weights = torch.load("model/model_d2.pth", map_location=device)

class PreProcessCamera:
    def __init__(self):
        self.os = self.get_os()
        self.path = self.get_camera_path()
    def get_os(self) ->cv2:
        """
        Autodetect which OS is being used  -> important for the detection of the camera name / index.
        Different os have different backend values to access for camera 
        """
        if platform.system() == 'Darwin':
            backend = cv2.CAP_AVFOUNDATION  #mac
        elif platform.system() == 'Windows':
            backend = cv2.CAP_MSMF #windows
        else:
            backend = cv2.CAP_V4L2 #linux
        return backend
    
    def get_camera_path(self) -> int:
        """
        Finding the correct camera to start the thread in the correct index
        FaceTime HD Camera ~ any macbook face camera
        """
        cams: list = enumerate_cameras(self.os) 
        for cam in cams:
            # Test the index before returning it
            if cam.name.lower() == "facetime hd camera": # -> change this in future sothat it works with the pi
                test_cap = cv2.VideoCapture(cam.index, self.os)
                if test_cap.isOpened():
                    success, _ = test_cap.read()
                    test_cap.release()
                    
                    if success:
                        print("CAMERA WORKING", cam.name, cam.index)
                        return cam.index
                    else:
                        print(f"Warning: Index {cam.index} matched name but failed to read a frame. Trying next...")
                else:
                    print(f"Warning: Index {cam.index} matched but failed to open.")

        raise ValueError("FaceTime HD Camera could not be found or opened for use.")
    
    

class Camera:
    def __init__(self, dropout_prob: float, model):
        super().__init__()
        self.load_model: LoadModel = LoadModel(dropout_prob)
        self.tensorizedframe: TensorizedFrame = TensorizedFrame()
        self.frame_counter = 0
        self.video = model.video
        self.model = model
       

       
    def get_video(self, output_queue ) -> bool:
        success, frame = self.video.read()
        if not success :
            raise ValueError ("Failed to verify video")

        self.frame_counter +=1
        if self.frame_counter % 3 ==0 and self.model.test_mode == True: # make system only work at 10 fps for now in order to not overload the cnn model
            correct_frame_format: torch.Tensor =  self.tensorizedframe.correct_tensor(frame)
            correct_frame_format = correct_frame_format.to(device)
            model_call: LoadModel = self.load_model.set_frame_to_model(correct_frame_format)
            prediction: tuple = self.load_model.get_predictions(model_call)
            output_queue.put(prediction)
        return True
    
 
    
    
    

class TensorizedFrame:

    def __init__(self) -> None:
        """This class has 1 major task which is to prepare the frame for the CNN"""
        self.corrected_frame = None
     

    
    def _corrected_cnn_format(self, frame: np.ndarray) -> np.ndarray:
        correct_frame_size: np.ndarray = cv2.resize(frame, (WIDTH,HEIGHT)) #my model was trained on 32 x 32 images so it is good to keep that format up 
        correct_format: np.ndarray = cv2.cvtColor(correct_frame_size, cv2.COLOR_BGR2RGB) # Tensors require RGB but cv2 outputs BGR meanuing ut must be converted
        return correct_format

    def _set_tensor_dimentions(self) -> torch.Tensor:
        tensor_frame = self.corrected_frame / 255.0
        tensor_frame: torch.Tensor = torch.tensor(tensor_frame).float()
        tensor_frame: torch.Tensor = tensor_frame.permute(2, 0, 1)
        
        tensor_frame: torch.Tensor = (tensor_frame -0.5)/0.5
        tensor_frame: torch.Tensor = tensor_frame.unsqueeze(0)
        return tensor_frame
    
    def correct_tensor(self, frame) -> torch.Tensor:
        self.corrected_frame: np.ndarray = self._corrected_cnn_format(frame)
        final_correct_tensor_format: torch.Tensor = self._set_tensor_dimentions()
        return final_correct_tensor_format

    

class LoadModel:
    def __init__(self, dropout_prob):
        self.model = model.CNN_model.SimpleCNN_dropout(dropout_prob).to(device)
        self.model.load_state_dict(trained_weights)# Here i have to add the finished trained weights
        self.model.to(device)
        self.model.eval()
        

    def set_frame_to_model(self,frame) -> model.CNN_model.SimpleCNN_dropout:
        with torch.no_grad():
            return self.model(frame)
    
    def get_predictions(self, model) -> tuple[str, float]:
        prediction_item = torch.argmax(model, dim = 1)
        probs = torch.softmax(model, dim = 1)
        pred_idx: int = prediction_item.item() #converts a pytorch tensor into a normal number
        confidence:float = probs[0][pred_idx].item()

        class_names = [
                "airplane", "automobile", "bird", "cat", "deer",
                "dog", "frog", "horse", "ship", "truck"]
        
        try:
            item_predicted: str = class_names[prediction_item]
            return (item_predicted, confidence)
        except IndexError:
            return False