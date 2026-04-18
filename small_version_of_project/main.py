
from sim_controller import SimController
class CallModel:
    def __init__(self):
        self.call_controller = SimController()
        if self.call_controller.system_setup() == False:
            raise ValueError("Failed to verify some part of the system")

    def run_model(self):
        camera = self.call_controller.call_model()
        while self.call_controller.running:
            self.call_controller.running_model(camera)
            

call_model = CallModel()
call_model.run_model()


#TODO -> update the UML Diagram to the needed standard
#TODO -> Figure out a way to make model more efficent

