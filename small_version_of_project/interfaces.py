from abc import ABC, abstractmethod
import torch
import cv2
import sys


class ICommands(ABC):
    
    @abstractmethod
    def execute(self) -> bool:
        pass

class IStartSystem(ABC):
    @abstractmethod
    def start_system(model) -> None:
        pass

class IEndSystem(ABC):
    @abstractmethod
    def end_system(model) -> None:
        pass
    @abstractmethod
    def terminate(video, pre_process) -> sys.exit:
        pass

class IStartTest(ABC):
    @abstractmethod
    def start_test(model):
        pass 

class IEndTest(ABC):
    @abstractmethod
    def end_test(model):
        pass

class IVideoShower(ABC):
    @abstractmethod
    def show_video(model):
        pass

    @abstractmethod
    def end_video(model):
        pass

class ITerminalManager(ABC):
    @abstractmethod
    def lock_terminal(model):
        pass

    @abstractmethod
    def release_terminal(model):
        pass





class IModelLoader(ABC):

    @abstractmethod
    def get_predictions(self, model) -> tuple[str, float]:
        pass

class ICameraPreper(ABC):
    
    @abstractmethod
    def get_camera_path(self) -> (str, int, bool):
        pass

    @abstractmethod
    def open_camera(self) -> cv2.VideoCapture:
        pass

class IFrameTensorizor(ABC):
    
    @abstractmethod
    def correct_tensor(self, frame) -> torch.tensor:
        pass

class ICamera(ABC):

    @abstractmethod
    def get_video(self, output_queue, show_recording) -> bool:
        pass



class ISimController(ABC):
     
    @abstractmethod
    def cameraprocesser(self) -> bool:
        pass

    @abstractmethod
    def model_loader(self) -> IModelLoader:
        pass

    @abstractmethod
    def model_runner(self, camera) -> None:
        pass

    @abstractmethod
    def tensorized_frame_loader(self) -> IFrameTensorizor:
        pass

    @abstractmethod
    def system_setup(self) -> bool:
        pass

    @abstractmethod
    def system_starter(self) -> None:
        pass

    