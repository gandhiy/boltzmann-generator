from abc import ABC, abstractmethod
import numpy as np

class Thermostat(ABC):
    def __init__(self, system, T):
        self.system = system
        self.T = T
    
    @abstractmethod
    def apply(self):
        pass


class ThermostatFactory():
    def __init__(self, system):
        self.system = system
        self.thermostats = {
                            "anderson" : AndersonThermostat,
                           }

    def get_thermostat(self, thermostat_name, T, **kwargs):
        if thermostat_name in self.thermostats.keys():
            return(self.thermostats[thermostat_name](self.system, T, **kwargs))
        else:
            raise NotImplementedError(thermostat_name + "is not a valid thermostat!")


class AndersonThermostat(Thermostat):
    def __init__(self, system, T, colisions =  0.01, freq = 50):
        super().__init__(system, T)
        self.freq = freq
        self.colisions = colisions

    def apply(self, steps):
        if steps % self.freq == 0:
            for particle in self.system.particles:
                if self.system.integrator.dt * self.freq * self.colisions > \
                     np.random.uniform():
                    particle.vel = np.random.normal(scale = np.sqrt(self.T/particle.mass),
                                                    size = particle.vel.shape)



            
    