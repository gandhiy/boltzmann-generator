import numpy as np

class WCAPotential:
    def __init__(self, sigma, epsilon):
        self.r_c = np.power(2, 1/6) * sigma 
        self.sigma = sigma
        self.epsilon = epsilon
        self.E_c = self.epsilon * 4 * ( (self.r_c / self.sigma) ** -12 - ( self.r_c / self.sigma) ** -6 )

    def __call__(self, r_ij):
        if np.dot(r_ij,r_ij) < self.r_c ** 2:
            return(self.epsilon * 4 * ( (self.sigma**2 / np.dot(r_ij, r_ij)**6) - (self.sigma**2 / np.dot(r_ij, r_ij)**3) ) - self.E_c )
        else:
            return(0.0)
    def derivative(self, r_ij):
        if np.dot(r_ij,r_ij) < self.r_c ** 2:
            return( 24 * self.epsilon * r_ij / np.dot(r_ij, r_ij) * (2 * (self.sigma**2 / np.dot(r_ij, r_ij)**6) - (self.sigma**2 / np.dot(r_ij, r_ij)**3) ) )
        else:
            return(np.zeros(r_ij.shape))

class LJpotential:
    def __init__(self, sigma, epsilon, r_c = None):
        self.r_c = r_c
        self.sigma = sigma
        self.epsilon = epsilon
        self.E_c = self.epsilon * 4 * ( (self.r_c / self.sigma) ** -12 - ( self.r_c / self.sigma) ** -6 )

    def __call__(self, r_ij):
        if np.dot(r_ij,r_ij) < self.r_c ** 2 or self.r_c is None:
            return(self.epsilon * 4 * ( (self.sigma**2 / np.dot(r_ij, r_ij)**6) - (self.sigma**2 / np.dot(r_ij, r_ij)**3) ))
        else:
            return(0.0)
    def derivative(self, r_ij):
        if np.dot(r_ij,r_ij) < self.r_c ** 2 or self.r_c is None:
            return( 24 * self.epsilon * r_ij / np.dot(r_ij, r_ij) * (2 * (self.sigma**2 / np.dot(r_ij, r_ij)**6) - (self.sigma**2 / np.dot(r_ij, r_ij)**3) ) )
        else:
            return(np.zeros(r_ij.shape))