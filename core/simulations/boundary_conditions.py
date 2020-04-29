from abc import ABC, abstractmethod
import numpy as np


class BoundaryCondition(ABC):
    def __init__(self, system):
        self.system = system

    @abstractmethod
    def __call__(self, coords):
        pass

        
        
class PeriodicBoundaryConditions(BoundaryCondition):
    """
    Class responsible for applying types of periodic boundary conditions


    Attributes
    ----------
    system : System
        system that is using the boundary conditions
    Method
    ------
    __call__(coords)
        apply boundary conditions to the coordinates
    """
    def __init__(self, system):
        super().__init__(system)

    def __call__(self, coords):
        return coords - self.system.box * np.round(coords / self.system.box)

class NoBoundaryConditions(BoundaryCondition):
    """
    Class responsible for applying no boundary conditions


    Attributes
    ----------
    system : System
        system that is using the boundary conditions
    Method
    ------
    __call__(coords)
        apply boundary conditions to the coordinates
    """
    def __init__(self, system):
        super().__init__(system)
    
    def __call__(self, coords):
        return coords


# Need to think how to implement this since velocities would also change
# Therefore different interface ... maybe adapter
# class ReflectiveBoundaryConditions(BoundaryCondition):
#     def __init__(self, system):
#         super().__init__(system)
# 
#     def __call__(self, coords):
#         return coords - self.system.box * (1 + np.round(coords / self.system.box))