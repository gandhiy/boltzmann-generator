from abc import ABC, abstractmethod
import numpy as np
import simulation_library
import tensorflow as tf
import data_logging
import potentials
import yaml

class SimulationData:
    def __init__(self, config_file):
        self.__simulation = None
        self.simulation_name = None
        self.simulation_data = None
        self.config_file = config_file
        self.sim_dict = {
            "muller-sim" : MuellerWellSim,
            "double-well-sim" : DoubleWellSim,
            "dimer-lj-sim" : DimerLJFluidSim,
        }
        with open(self.config_file, 'r') as fs:
            self.config = yaml.safe_load(fs)
        self.simulation = self.sim_dict[self.config["simulation_name"]](self.config_file)
        self.simulation_name = self.config["simulation_name"]

    def loadSimulation(self, data_file):
        self.simulation_data = np.load(data_file)

    def saveSimulation(self, data_file):
        np.save(data_file, self.simulation_data)

    def getSimulation(self):
        return self.simulation

    def runSimulation(self):
        self.simulation.runSimulation()

    def getData(self):
        if self.simulation_data is None:
            self.simulation_data = np.array(self.simulation.coordinate_logger.coordinates)
        return self.simulation_data

    def getEnergy(self, coords = None):
        return self.simulation.getEnergy(coords)

class Simulation(ABC):
    def __init__(self, config_file):
        self.config_file = config_file
        with open(self.config_file, 'r') as fs:
            self.config = yaml.safe_load(fs)

    @abstractmethod
    def runSimulation(self):
        pass

    def getEnergy(self, coords):
        pass

    def getEnergy_tf(self, coords):
        pass

    def getData(self):
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
        self.system.central_potential = self.central_potential
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

    def getEnergy(self, coords = None):
        if coords is not None:
            self.system.set_coordinates(coords)
        return self.system.get_energy()[1]

    def getEnergy_tf(self, coords):
        self.system.set_coordinates_tf(coords)
        return self.system.get_energy_tf()[1]

    def getData(self):
        return(np.array(self.coordinate_logger.coordinates))

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
        self.system.central_potential = self.central_potential
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

    def getEnergy(self, coords = None):
        if coords is not None:
            self.system.set_coordinates(coords)
        return self.system.get_energy()[1]

    def getEnergy_tf(self, coords):
        self.system.set_coordinates_tf(coords)
        return self.system.get_energy_tf()[1]

    def getData(self):
        return(np.array(self.coordinate_logger.coordinates))

class DimerLJFluidSim(Simulation):
    def __init__(self, config_file):
        super().__init__(config_file)
        self.system_builder = simulation_library.SystemFactory()
        self.system = self.system_builder.build_system(dim = 2, **self.config["system_params"])
        self.system.add_bond(potentials.DoubleWellPotential1D(1.5, 25, 10, -0.5), self.config["bond_indexes"][0], self.config["bond_indexes"][1])
        self.system.bonds[0].particle_interactions = False
        self.system.central_potential = potentials.HarmonicBox(l_box = 3.0, k_box = 100)
        self.system.get_integrator(self.config["integrator"], **self.config["integrator_args"])
        if "thermostat" in self.config.keys():
            self.system.get_thermostat(self.config["thermostat"], **self.config["thermostat_args"])
        self.coordinate_logger = data_logging.CoordinateLogger(self.system, self.config["out_freq"])
        self.energy_logger = data_logging.EnergyLogger(self.system, self.config["out_freq"])
        self.dist_logger = data_logging.DistanceLogger(self.system, 100)
        self.system.registerObserver(self.coordinate_logger)
        self.system.registerObserver(self.energy_logger)
        self.system.registerObserver(self.dist_logger)

    def runSimulation(self):
        self.system.run(self.config["n_steps"])

    def getEnergy(self, coords = None):
        if coords is not None:
            self.system.set_coordinates(coords)
        return self.system.get_energy()[1]

    def getEnergy_tf(self, coords):
        self.system.set_coordinates_tf(coords)
        return self.system.get_energy_tf()[1]
        
    def getData(self):
        return(np.array(self.coordinate_logger.coordinates))
