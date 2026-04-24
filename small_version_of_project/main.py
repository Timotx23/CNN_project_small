
from sim_controller import SimController
from interfaces import ISimController



if __name__ == '__main__':
    call_controller: ISimController  = SimController()
    call_controller.system_starter()


#TODO -> Add a system so that u can select which camera is going to be used 
#TODO -> update the UML Diagram to the needed standard
#TODO -> Figure out a way to make model more efficent
 


