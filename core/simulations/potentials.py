from abc import ABC, abstractclassmethod
import numpy as np

class Potential(ABC):
    @abstractclassmethod
    def __call__(self, rij):
        pass

    def derivative(self, rij):
        pass


# Particle Potentials
class WCAPotential:
    """
    Object representing  WCA potential

    Attributes
    ----------
    r_c : float
        cutoff distance
    sigma : float
        sigma parameter
    epsilon : float
        epsilon paramter
    E_c : foat
        Energy at the cutoff distance
    
    Methods
    -------
    __call__(r_ij)
        calling the potential energy class returns the energy
        of the provided vector
    derivative(r_ij)
        returns the derivative of the potential at the provided
        vector
    """
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
            return( -24 * self.epsilon * r_ij / np.dot(r_ij, r_ij) * (2 * (self.sigma**2 / np.dot(r_ij, r_ij)**6) - (self.sigma**2 / np.dot(r_ij, r_ij)**3) ) )
        else:
            return(np.zeros(r_ij.shape))

class LJpotential:
    """
    Object representing  LJ potential

    Attributes
    ----------
    r_c : float
        cutoff distance
    sigma : float
        sigma parameter
    epsilon : float
        epsilon paramter
    
    Methods
    -------
    __call__(r_ij)
        calling the potential energy class returns the energy
        of the provided vector
    derivative(r_ij)
        returns the derivative of the potential at the provided
        vector
    """
    def __init__(self, sigma, epsilon, r_c = None):
        self.r_c = r_c
        self.sigma = sigma
        self.epsilon = epsilon

    def __call__(self, r_ij):
        if np.dot(r_ij,r_ij) < self.r_c ** 2 or self.r_c is None:
            return(self.epsilon * 4 * ( (self.sigma**2 / np.dot(r_ij, r_ij)**6) - (self.sigma**2 / np.dot(r_ij, r_ij)**3) ))
        else:
            return(0.0)

    def derivative(self, r_ij):
        if np.dot(r_ij,r_ij) < self.r_c ** 2 or self.r_c is None:
            return( -24 * self.epsilon * r_ij / np.dot(r_ij, r_ij) * (2 * (self.sigma**2 / np.dot(r_ij, r_ij)**6) - (self.sigma**2 / np.dot(r_ij, r_ij)**3) ) )
        else:
            return(np.zeros(r_ij.shape))

class LJRepulsion:
    """
    Object representing the repulive contribution of the LJ potential

    Attributes
    ----------
    r_c : float
        cutoff distance
    sigma : float
        sigma parameter
    epsilon : float
        epsilon paramter
    
    Methods
    -------
    __call__(r_ij)
        calling the potential energy class returns the energy
        of the provided vector
    derivative(r_ij)
        returns the derivative of the potential at the provided
        vector
    """
    def __init__(self, sigma, epsilon, r_c = None):
        self.r_c = np.power(2, 1/6) * sigma 
        self.sigma = sigma
        self.epsilon = epsilon

    def __call__(self, r_ij):
        if np.dot(r_ij,r_ij) < self.r_c ** 2 or self.r_c is None:
            return(self.epsilon * 4 * ((self.sigma**2 / np.dot(r_ij, r_ij)**6) ))
        else:
            return(0.0)

    def derivative(self, r_ij):
        if np.dot(r_ij,r_ij) < self.r_c ** 2 or self.r_c is None:
            return( -48 * self.epsilon *  self.sigma * r_ij * (1 / np.dot(r_ij, r_ij)**7))
        else:
            return(np.zeros(r_ij.shape))

# 1D Potentials (for Internal Coordinates)
class DoubleWellPotential1D:
    """
    Object representing a 1D doubler well potential

    V(x) = 1/4 a (x - d0) ^ 4 - 1/2 b (x - d0) ^ 2 + c (x - d0)

    Attributes
    ----------
    d0 : float
        offset of the potential from the origin
    a : float
        a parameter
    b : float
        b parameter
    c : float
        c parameter
    
    Methods
    -------
    __call__(r_ij)
        calling the potential energy class returns the energy
        of the provided vector
    derivative(r_ij)
        returns the derivative of the potential at the provided
        vector
    """
    def __init__(self, d0, a, b, c):
        self.d0 = d0
        self.a = a
        self.b = b
        self.c = c

    def __call__(self, d):
        return 1/4 * self.a * (d - self.d0) ** 4 - 1/2 * self.b * (d - self.d0) ** 2 + \
            self.c * (d - self.d0)

    def derivative(self, d):
        return self.a * (d - self.d0) ** 3 - self.b * (d - self.d0) + self.c


# Central Potentials
class DoubleWellPotential:
    """
    Object representing a 2D double well potential

    V(x) = 1/4 a x ^ 4 - 1/2 b x ^ 2 + c x  + d y ^ 2

    Attributes
    ----------
    a : float
        a parameter
    b : float
        b parameter
    c : float
        c parameter
    d : float
        d parameter
    
    Methods
    -------
    __call__(r_ij)
        calling the potential energy class returns the energy
        of the provided vector
    derivative(r_ij)
        returns the derivative of the potential at the provided
        vector
    """
    def __init__(self, a, b, c, d):
        self.a = a
        self.b = b
        self.c = c
        self.d = d
    
    def __call__(self, r):
        return 1/4 * self.a * r[0] ** 4 - 1/2 * self.b * r[0] ** 2 + \
            self.c * r[0] + 1/2 * self.d * r[1] ** 2

    def derivative(self, r):
        dx = self.a * r[0] ** 3 - self.b * r[0] + self.c
        dy = self.d * r[1]
        return np.array([dx, dy])

class HarmonicPotential:
    """
    Object representing a harmonic well potential

    V(x) = 1/2 k * (x - x_0)^2

    Attributes
    ----------
    k : float
        spring constant
    x_o : np.ndarray
        resting position

    
    Methods
    -------
    __call__(r_ij)
        calling the potential energy class returns the energy
        of the provided vector
    derivative(r_ij)
        returns the derivative of the potential at the provided
        vector
    """
    def __init__(self, k, x_o = None):
        self.k = k
        if x_o is None:
            self.x_o = 0
        else:
            self.x_o = x_o
    
    def __call__(self, r):
        return 1/2 * self.k * np.dot((r - self.x_o), (r - self.x_o))

    def derivative(self, r):
        return self.k * (r - self.x_o)

class MuellerPotential:
    """
    Object representing a Muller-Brown potential surface.
    The surface can be thought of as the sum of several
    2D gaussians

    Attributes
    ----------
    A : list
        list of A parameters
    a : list
        list of a parameters
    b : list
        list of b parameters
    c : list
        list of c parameters
    d : list
        list of d parameters
    xj : list
        list of x position of gaussian centers
    yj : list
        list of y position of gaussian centers
    
    Methods
    -------
    __call__(r_ij)
        calling the potential energy class returns the energy
        of the provided vector
    derivative(r_ij)
        returns the derivative of the potential at the provided
        vector
    """
    def __init__(self, alpha, A, a, b, c, xj, yj):
        self.alpha = alpha
        self.A = A
        self.a = a
        self.b = b
        self.c = c
        self.xj = xj
        self.yj = yj
    
    def __call__(self, r):
        E = 0
        i = 0
        for A, a, b, c, xj, yj in zip(self.A, self.a, self.b, self.c, self.xj, self.yj):
            i += 1
            E += A * np.exp(a * (r[0] - xj)**2 + b * (r[0] - xj) * (r[1] - yj) + c * (r[1] - yj)**2)
        return(self.alpha * E)

    def derivative(self, r):
        dx = 0
        dy = 0
        for A, a, b, c, xj, yj in zip(self.A, self.a, self.b, self.c, self.xj, self.yj):
            dx += A * np.exp(a * (r[0] - xj)**2 + b * (r[0] - xj) * (r[1] - yj) + c * (r[1] - yj)**2) * (2 * a * (r[0] - xj) + b * (r[1] - yj))
            dy += A * np.exp(a * (r[0] - xj)**2 + b * (r[0] - xj) * (r[1] - yj) + c * (r[1] - yj)**2) * (b * (r[0] - xj) + 2 * c * (r[1] - yj))
        # print("dE =", self.alpha * np.array([dx, dy]))
        # print("r = ", r)
        return(self.alpha * np.array([dx, dy]))

class HarmonicBox:
    """
    Object representing a Muller-Brown potential surface.
    The surface can be thought of as the sum of several
    2D gaussians

    Attributes
    ----------
    l_box : np.ndarray
        box size
    k_box : float
        harmonic potential outside of box
    Methods
    -------
    __call__(r_ij)
        calling the potential energy class returns the energy
        of the provided vector
    derivative(r_ij)
        returns the derivative of the potential at the provided
        vector
    """
    def __init__(self, l_box, k_box):
        self.l_box = l_box
        self.k_box = k_box

    def __call__(self, r_ij):
        u_upper = np.sum(np.heaviside(r_ij - self.l_box/2, 0) * self.k_box * (r_ij - self.l_box/2) ** 2)
        u_lower = np.sum(np.heaviside(-r_ij - self.l_box/2, 0) * self.k_box * (-r_ij - self.l_box/2) ** 2)
        return u_upper + u_lower

    def derivative(self, r_ij):
        du_upper = np.heaviside(r_ij - self.l_box/2, 0) * 2 * self.k_box * (r_ij - self.l_box/2)
        du_lower = np.heaviside(-r_ij - self.l_box/2, 0) * -2 * self.k_box * (-r_ij - self.l_box/2)
        return du_upper + du_lower

