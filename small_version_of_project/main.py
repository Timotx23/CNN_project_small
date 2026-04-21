
from sim_controller import SimController
from interfaces import ISimController
class CallSim:
    def __init__(self):
        self.call_controller: ISimController  = SimController()
        if self.call_controller.system_setup() == False:
            raise ValueError("Failed to verify some part of the system")

    def run_model(self):
        camera = self.call_controller.call_model()
        while self.call_controller.running:
            self.call_controller.running_model(camera)
            

call_sim = CallSim()
call_sim.run_model()

#TODO -> Add a system so that u can select which camera is going to be used 
#TODO -> update the UML Diagram to the needed standard
#TODO -> Figure out a way to make model more efficent
#TODO -> Fix the naming scheme at some points 
#TODO -> Rely more heavily on interfaces in order to make the project more scalable 

