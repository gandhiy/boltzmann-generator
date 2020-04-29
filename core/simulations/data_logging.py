from abc import ABC, abstractmethod
import numpy as np

class Subject(ABC):
    def __init__(self):
        self.observers = []

    @abstractmethod
    def registerObserver(self, Observer):
        pass

    @abstractmethod
    def removeObserver(self, Observer):
        pass
    
    @abstractmethod
    def notifyObservers(self):
        pass



class Observer(ABC):
    @abstractmethod
    def update(self):
        pass


class CoordinateLogger(Observer):
    """
    Object responsible for saving simulation coordinates

    Attributes
    ----------
    system : System
        system the logger is watching
    coordinates : list
        list of accumulated coordinates
    step : list
        list of accumulated steps
    freq : int
        frequency to write to logger
    """
    def __init__(self, system, freq):
        self.system = system
        self.coordinates = []
        self.steps = []
        self.freq = freq
        self.update(0)

    def update(self, steps):
        if len(self.steps) != 0:
            if steps < max(self.steps):
                steps += max(self.steps)
        if steps % self.freq == 0:
            self.steps.append(steps)
            self.coordinates.append(self.system.get_coordinates())
    

class EnergyLogger(Observer):
    """
    Object responsible for saving simulation energy

    Attributes
    ----------
    system : System
        system the logger is watching
    K : list
        list of accumulated kinetic energy values
    U : list
        list of accumulated potential energy values
    H : list
        list of accumulated total energy values
    step : list
        list of accumulated steps
    freq : int
        frequency to write to logger
    """
    def __init__(self, system, freq):
        self.system = system
        self.K = []
        self.U = []
        self.H = []
        self.steps = []
        self.freq = freq
        self.update(0)

    def update(self, steps):
        if len(self.steps) != 0:
            if steps < max(self.steps):
                steps += max(self.steps)
        if steps % self.freq == 0:
            self.steps.append(steps)
            h, u, k = self.system.integrator.calculate_energy()
            self.K.append(k)
            self.U.append(u)
            self.H.append(h)

class PressureLogger(Observer):
    """
    Object responsible for saving simulation pressure

    Attributes
    ----------
    system : System
        system the logger is watching
    pressure : list
        list of accumulated pressure values
    step : list
        list of accumulated steps
    freq : int
        frequency to write to logger
    """
    def __init__(self, system, freq):
        self.system = system
        self.P = []
        self.steps = []
        self.freq = freq
        self.update(0)

    def update(self, steps):
        if len(self.steps) != 0:
            if steps < max(self.steps):
                steps += max(self.steps)
        if steps % self.freq == 0:
            self.steps.append(steps)
            p = self.system.integrator.calculate_pressure()
            self.P.append(p)

class VelocityLogger(Observer):
    """
    Object responsible for saving simulation velocities

    Attributes
    ----------
    system : System
        system the logger is watching
    velocities : list
        list of accumulated velocities values
    step : list
        list of accumulated steps
    freq : int
        frequency to write to logger
    """
    def __init__(self, system, freq):
        self.system = system
        self.vels = []
        self.steps = []
        self.freq = freq
        self.update(0)

    def update(self, steps):
        if len(self.steps) != 0:
            if steps < max(self.steps):
                steps += max(self.steps)
        if steps % self.freq == 0:
            self.steps.append(steps)
            v = self.system.get_velocities()
            self.vels.append(v)

class TemperatureLogger(Observer):
    """
    Object responsible for saving simulation temperature

    Attributes
    ----------
    system : System
        system the logger is watching
    temperature : list
        list of accumulated temperature values
    step : list
        list of accumulated steps
    freq : int
        frequency to write to logger
    """
    def __init__(self, system, freq):
        self.system = system
        self.T = []
        self.steps = []
        self.freq = freq
        self.update(0)

    def update(self, steps):
        if len(self.steps) != 0:
            if steps < max(self.steps):
                steps += max(self.steps)
        if steps % self.freq == 0:
            self.steps.append(steps)
            T = self.system.integrator.calculate_temperature()
            self.T.append(T)

class DistanceLogger(Observer):
    """
    Object responsible for saving distances of interest over the simulation

    Attributes
    ----------
    system : System
        system the logger is watching
    d : list
        list of accumulated energy distance values
    step : list
        list of accumulated steps
    freq : int
        frequency to write to logger
    """
    def __init__(self, system, freq, bonds = None):
        if bonds is None:
            bonds = system.bonds
        self.bonds = bonds
        self.system = system
        self.d = []
        self.steps = []
        self.freq = freq
        self.update(0)

    def update(self, steps):
        if len(self.steps) != 0:
            if steps < max(self.steps):
                steps += max(self.steps)
        if steps % self.freq == 0:
            self.steps.append(steps)
            d_s = np.array([bond.get_distance() for bond in self.bonds])
            self.d.append(d_s)
