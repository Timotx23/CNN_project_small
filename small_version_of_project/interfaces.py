from abc import ABC, abstractmethod

class ICommands(ABC):
    
    @abstractmethod
    def execute(self):
        pass

class IStartSystem(ABC):
    @abstractmethod
    def start_system(model):
        pass

class IEndSystem(ABC):
    @abstractmethod
    def end_system(model):
        pass
    @abstractmethod
    def terminate(video):
        pass

class IStartTest(ABC):
    @abstractmethod
    def start_test(model):
        pass 

class IEndTest(ABC):
    @abstractmethod
    def end_test(model):
        pass
class IShowVideo(ABC):
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


class ISimController(ABC):
    @abstractmethod
    def call_model(self):
        pass
    @abstractmethod
    def camera_preprocessing(self):
        pass

    @abstractmethod
    def load_model(self):
        pass

    @abstractmethod
    def running_model(self, camera):
        pass

    @abstractmethod
    def load_tensorized_frame(self):
        pass

    @abstractmethod
    def system_setup(self):
        pass