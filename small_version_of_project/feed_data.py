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
        self.system = platform.system()
        self.backend = self.get_os()
        self.camera_type, self.path, self.cv_backend = self.get_camera_path()

    def get_os(self) -> int:
        """
        Detect OS-specific OpenCV backend.
        """
        if self.system == "Darwin":
            return cv2.CAP_AVFOUNDATION
        elif self.system == "Windows":
            return cv2.CAP_MSMF
        else:
            return cv2.CAP_V4L2

    def get_camera_path(self):
        """
        Returns:
            ("picamera2", 0, None) for Raspberry Pi CSI camera
            ("cv2", cam_index, cam_backend) for regular webcams
        """

        # Raspberry Pi CSI camera path
        if self.system == "Linux":
            try:
                from picamera2 import Picamera2

                picam2 = Picamera2(0)
                picam2.configure(
                    picam2.create_preview_configuration(main={"size": (640, 480)})
                )
                picam2.start()
                frame = picam2.capture_array()
                picam2.stop()

                if frame is not None:
                    print("CAMERA WORKING imx219 0")
                    return "picamera2", 0, None

            except Exception as e:
                print(f"Picamera2 test failed: {e}")

        # Fallback for normal OpenCV webcams
        cams = enumerate_cameras(self.backend)

        for cam in cams:
            cam_name = cam.name.lower()

            if "facetime" in cam_name or "webcam" in cam_name or "camera" in cam_name:
                test_cap = cv2.VideoCapture(cam.index, cam.backend)

                if test_cap.isOpened():
                    success, _ = test_cap.read()
                    test_cap.release()

                    if success:
                        print("CAMERA WORKING", cam.name, cam.index)
                        return "cv2", cam.index, cam.backend
                    else:
                        print(f"Warning: Index {cam.index} matched name but failed to read a frame.")
                else:
                    print(f"Warning: Index {cam.index} matched but failed to open.")

        raise ValueError("No usable camera could be found.")

    def open_camera(self):
        """
        Open camera based on detected backend.
        """
        if self.camera_type == "picamera2":
            from picamera2 import Picamera2

            cam = Picamera2(self.path)
            cam.configure(
                cam.create_preview_configuration(main={"size": (640, 480)})
            )
            cam.start()
            return cam

        elif self.camera_type == "cv2":
            cam = cv2.VideoCapture(self.path, self.cv_backend)
            if not cam.isOpened():
                raise ValueError("Failed to open OpenCV camera.")
            return cam

        raise ValueError("Unknown camera type.")


class Camera:
    def __init__(self, dropout_prob: float, model):
        super().__init__()
        self.load_model: LoadModel = LoadModel(dropout_prob)
        self.tensorizedframe: TensorizedFrame = TensorizedFrame()
        self.frame_counter = 0
        self.model = model

        self.pre_process_camera = self.model.pre_process_camera
        self.video = self.model.video

    def read_frame(self):
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

    def get_video(self, output_queue) -> bool:
        success, frame = self.read_frame()

        if not success or frame is None:
            raise ValueError("Failed to verify video")

        self.frame_counter += 1

        if self.frame_counter % 3 == 0 and self.model.test_mode is True:
            correct_frame_format: torch.Tensor = self.tensorizedframe.correct_tensor(frame)
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