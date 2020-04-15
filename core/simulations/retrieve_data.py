from abc import ABC, abstractmethod
import numpy as np
import simulations

class RetrieveData(ABC):
    def __init__(self, file):
        self.file = file

def readData(self):
    pass



class ReadCSV(RetrieveData):
    pass

class SimulationAdapter(RetrieveData):
    pass

class ReadPDB(RetrieveData):