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

class ITerminalManger(ABC):
    @abstractmethod
    def lock_terminal(model):
        pass

    @abstractmethod
    def release_terminal(model):
        pass