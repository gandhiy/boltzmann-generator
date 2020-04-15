from abc import ABC, abstractmethod
import numpy as np
import simulation_library
import data_logging
import potentials
import yaml

class Simulation(ABC):
    def __init__(self, config_file):
        self.config_file = config_file
        with open(self.config_file, 'r') as fs:
            self.config = yaml.safe_load(fs)

    @abstractmethod
    def runSimulation(self):
        pass

class MuellerWellSim(Simulation):
    def __init__(self, config_file):
        super().__init__(config_file)
        self.system = simulation_library.System(dim = 2)
        self.central_potential = potentials.MuellerPotential(alpha = 0.1,
                             A = [-200, -100, -170, 15],
                             a = [-1, -1, -6.5, 0.7],
                             b = [0, 0, 11, 0.6],
                             c = [-10, -10, -6.5, 0.7],
                             xj = [1, 0, -0.5, -1],
                             yj = [0, 0.5, 1.5, 1]
                            )
        self.system.add_particle(simulation_library.Particle(None, np.array(self.config["x_0"])))
        self.system.get_integrator(self.config["integrator"], **self.config["integrator_args"])
        if "thermostat" in self.config.keys():
            self.system.get_thermostat(self.config["thermostat"], **self.config["thermostat_args"])
        self.coordinate_logger = data_logging.CoordinateLogger(self.system, self.config["out_freq"])
        self.energy_logger = data_logging.EnergyLogger(self.system, self.config["out_freq"])
        self.system.registerObserver(self.coordinate_logger)
        self.system.registerObserver(self.energy_logger)
    
    def runSimulation(self):
        self.system.run(self.config["n_steps"])

class DoubleWellSim(Simulation):
    def __init__(self, config_file):
        super().__init__(config_file)
        self.system = simulation_library.System(dim = 2)
        self.central_potential = potentials.\
            DoubleWellPotential(a = 1,
                                b = 6,
                                c = 1,
                                d = 1,
                                )
        self.system.add_particle(simulation_library.Particle(None, np.array(self.config["x_0"])))
        self.system.get_integrator(self.config["integrator"], **self.config["integrator_args"])
        if "thermostat" in self.config.keys():
            self.system.get_thermostat(self.config["thermostat"], **self.config["thermostat_args"])
        self.coordinate_logger = data_logging.CoordinateLogger(self.system, self.config["out_freq"])
        self.energy_logger = data_logging.EnergyLogger(self.system, self.config["out_freq"])
        self.system.registerObserver(self.coordinate_logger)
        self.system.registerObserver(self.energy_logger)
    
    def runSimulation(self):
        self.system.run(self.config["n_steps"])



