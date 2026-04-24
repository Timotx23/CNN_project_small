import cv2
import torch
import numpy as np
import model.CNN_model
from interfaces import IModelLoader, ICameraPreper, IFrameTensorizor, ICamera, ISimController, IFrameTensorizor, IModel


class Camera(ICamera):
    def __init__(self,model):
        
        self.model: ISimController = model
        self.CNN_model = model.model_loader()

        self.load_model: IModelLoader = self.CNN_model# fix
        
        self.frame_counter = 0
       
        self.pre_process_camera: ICameraPreper = self.model.pre_process_camera
        self.video: ICameraPreper.open_camera = self.model.video

    def _read_frame(self) -> (bool, cv2 ):
        """
        Read one frame from either Picamera2 or OpenCV.
        """
        if self.pre_process_camera.camera_type == "picamera2":
            frame = self.video.capture_array()
            frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
            return True, frame

        elif self.pre_process_camera.camera_type == "cv2":
            return self.video.read()

        return False, None

    def get_video(self, output_queue, show_recording) -> bool:
        success, frame = self._read_frame()

        if not success or frame is None:
            raise ValueError("Failed to verify video")

        self.frame_counter += 1
        if show_recording == True:
            cv2.imshow("Camera feed" , frame)
            cv2.waitKey(1)

        if self.frame_counter % 3 == 0 and self.model.test_mode is True:
            model_predictions: IModelLoader.get_predictions = self.CNN_model.model_predictions(frame)
            if self.model.terminal_mode == "model":
                output_queue.put(model_predictions)
        return True
    
    
    
class Model(IModel):
    def __init__(self, height, width, rgb, device, dropout_prob,trained_weights):
        self.tensor_frame = FrameTensorizor(height, width, rgb, device )
        self.model_loader = ModelLoader(dropout_prob, device, trained_weights)
    

    def _tensorized_frame(self, frame):
        """Ensures that the frame that is being collected is correct with the data on which the CNN was trained on. It takes a frame and turns it into the correct tensor which is passed onto the model"""
        return self.tensor_frame.correct_tensor(frame)
    
    def model_predictions(self, frame):
        correct_frame_format = self._tensorized_frame(frame)
        return self.model_loader.get_predictions(correct_frame_format)

    
class FrameTensorizor(IFrameTensorizor):

    def __init__(self, height, width, rgb, device) -> None:
        """This class has 1 major task which is to prepare the frame for the CNN
        This function can still be improved for better data feed for the CNN
        """
        self.device = device
        self.corrected_frame = None
        self.height = height
        self.width = width
        self.rgb =rgb
    def _corrected_cnn_format(self, frame: np.ndarray) -> np.ndarray:
        correct_frame_size: np.ndarray = cv2.resize(frame, (self.width,self.height)) #my model was trained on 32 x 32 images so it is good to keep that format up 
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
        final_correct_tensor_format = final_correct_tensor_format.to(self.device)
        return final_correct_tensor_format

    

class ModelLoader(IModelLoader):
    def __init__(self, dropout_prob, device, trained_weights):

        try:
            self.model = model.CNN_model.SimpleCNNDropout(dropout_prob).to(device)
            self.model.load_state_dict(trained_weights) # Here i have to add the finished trained weights
            self.model.to(device)
            self.model.eval()
            self.frame = None
        except:
            return False
        

    def _send_frame_to_model(self,frame) -> model.CNN_model.SimpleCNNDropout:
        with torch.no_grad():
            return self.model(frame)
    
    def get_predictions(self, frame) -> tuple[str, float]:
        model_processed_frame = self._send_frame_to_model(frame)
        prediction_item = torch.argmax(model_processed_frame, dim = 1)
        probs = torch.softmax(model_processed_frame, dim = 1)
        pred_idx: int = prediction_item.item() #converts a pytorch tensor into a normal number
        confidence:float = probs[0][pred_idx].item()

        class_names = [
                "airplane", "automobile", "bird", "cat", "deer",
                "dog", "frog", "horse", "ship", "truck"]
        
        try:
            item_predicted: str = class_names[pred_idx]
            return (item_predicted, confidence)
        except IndexError:
            return False