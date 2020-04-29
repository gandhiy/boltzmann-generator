import numpy as np
import integrators
import thermostats
import itertools
import potentials
import data_logging
import boundary_conditions
class Particle:
    """
    Particle object stores data about a particle

    Attributes
    ----------
    potential : Potential
        interaction potential the particle uses when interacting with other particles
    loc : np.ndarray
        location in ND space
    mass : float
        particle mass
    vel : np.ndarray
        ND velocity of particle
    force : np.ndaarray
        stored sum of all forces on particle
    neighbors : list
        neighbor list for neighborlist verlet integration
    cell : int
        cell number particle is in for cell list verlet integrator
    index : int
        index of cell in system.particle list
    """
    def __init__(self, potential, loc, vel = None, mass = 1):
        self.potential = potential
        self.loc = loc
        self.dim = len(loc)
        self.mass = mass
        self.vel = vel
        self.force = None
        self.neighbors = []
        self.prev_loc = None # <---- maybe implement as a particle decorator
        self.cell = None # <---- same thing here
        self.index = None

class Bond:
    """
    The Bond object stores data about the bond between two particles

    Attributes
    ----------
    particle_1 : Particle
        first particle in bond
    particle_2 : Particle
        second particle in bond
    potential : Potential
        interaction potential along bond
    bc : BoundaryCondition
        Boundary condition on how to apply bond accross box boundaries

    Methods
    -------
    get_distance()
        calculate the scalar distance between both particles
    get_rij()
        calcualte the vector distance between both particles
    get_energy()
        calculate the energy of the bond between both particles
    get_force()
        calculate the force exerted between both particles
    """
    def __init__(self, potential, particle_1, particle_2, particle_interactions = False, bc = None):
        self.particle_1 = particle_1
        self.particle_2 = particle_2
        self.potential = potential
        self.particle_interactions = particle_interactions
        self.bc = bc
        if bc is None:
            self.bc = boundary_conditions.NoBoundaryConditions(None)


    def get_distance(self):
        r_ij = self.bc(self.particle_1.loc - self.particle_2.loc)
        return(np.sqrt(np.dot(r_ij,r_ij)))

    def get_rij(self):
        return self.bc(self.particle_1.loc - self.particle_2.loc)
    
    def get_energy(self):
        d_ij = self.get_distance()
        return self.potential(d_ij)

    def get_force(self):
        d_ij = self.get_distance()
        f = -self.potential.derivative(d_ij)
        r_ij_hat = self.bc(self.particle_1.loc - self.particle_2.loc) / d_ij
        f_ij = f * r_ij_hat
        return f_ij


class SystemFactory:
    """
    Factory object for building system objects

    Attributes
    ----------
    placement : dict
        dictionary containing placement methods
    potentials : dict
        dictonary containing different particle potentials
    central_potentials : dict
        dictonary containing different central potentials
    boundary_conditions : dict
        dictionary containing different boundary conditions
    
    Methods
    -------
    build_system(self, dim = 2, T = 1, rho = 0.6, N = 20, mass = 1,
                 placement_method = "lattice",
                 boundary_conditions = "periodic",
                 potential = "WCA", **args)
        method for building a system object with many
        different options regarind particle placement, 
        system temperature, system density, etc.
    get_dof(system, boundary_conditions)
        set the number of dofs for a system
    add_central_potential(system, central_potential, **kwargs)
        add a central potential to the system
    lattice_placement(n, box)
        place particles in box using a lattice placement method
    random_placement(n, box, dmin=None, center=None)
        place particles randomly in box. if dmin is specified particles
        will be placed such that no particles are within dmin of each
        other. If center is specified, it changes the center of where
        particles are added.
    """
    def __init__(self):
        self.placement = {
                          "lattice" : self.lattice_placement,
                          "random" : self.random_placement,
                         }
        
        self.potentials = {
                           "WCA" : potentials.WCAPotential,
                           "LJ" : potentials.LJpotential,
                           "LJ-rep" : potentials.LJRepulsion
                          }

        self.central_potentials = {
                                   "harmonic" : potentials.HarmonicPotential,
                                   "double_well" : potentials.DoubleWellPotential,
                                   "mueller" : potentials.MuellerPotential
                                  }

        self.boundary_conditions = {
                                    "periodic" : boundary_conditions.PeriodicBoundaryConditions,
                                    "none" : boundary_conditions.NoBoundaryConditions
                                   }


    def build_system(self, 
                    dim = 2,
                    T = 1,
                    rho = 0.6,
                    N = 20,
                    mass = 1,
                    placement_method = "lattice",
                    boundary_conditions = "periodic",
                    potential = "WCA",
                    **args):
        if len(args) == 0:
            args = {"sigma" : 1, "epsilon" : 1}   
        box = (N / rho) ** (1/dim) * np.ones(dim)
        coords = self.placement[placement_method](N, box)
        system = System(box=box)
        system.bc = self.boundary_conditions[boundary_conditions](system)
        vels = np.random.normal(scale = np.sqrt(T/mass), size = (N, dim))
        for coord, vel in zip(coords, vels):
            system.add_particle(Particle(self.potentials[potential](**args), coord, vel = vel, mass = mass))
        self.get_dofs(system, placement_method)
        return system

    def get_dofs(self, system, boundary_conditions):
        system.dof = system.dim * len(system.particles)
        if boundary_conditions == "periodic":
            system.dof -= system.dim
        
    def add_central_potential(self, system, central_potential, **kwargs):
        system.central_potential = self.central_potentials[central_potential](**kwargs)
    
    def lattice_placement(self, n, box):
        r_min = -box / 2
        r_max = - r_min
        dim = len(box)
        d = box / (n)**(1/dim)
        pos = np.linspace(r_min + 0.5 * d, r_max - 0.5 * d, np.ceil(n**(1/dim)))
        coords = np.array(list(itertools.product(pos, repeat=dim)))
        coords = np.array([np.diagonal(coord) for coord in coords])
        return coords

    def random_placement(self, n, box, d_min = None, center = None):
        dim = len(box)
        if d_min is None:
            d_min = 0.1 * np.average(box)
            print(d_min)
        if center is None:
            center = np.zeros(dim)
        coords = np.zeros((n, dim))
        coords[0] = (np.random.random(dim) - 0.5)*box + center
        i = 0
        while i <= n - 1:
            prop = (np.random.random(dim) - 0.5) * box + center
            if np.any(np.linalg.norm(prop - coords, axis=1) > d_min):
                coords[i] = prop
                i += 1
        print(coords)
        return coords


class System(data_logging.Subject):
    """
    System object stores the state of a system while the simulation is running.
    The system object is only responsible for the instantaneous configuration 
    of the system as it is propagated through time.

    Attributes
    ----------
    observers : list
        list of observers observing the system
    particles : list
        list of Particle objects in the system
    bonds : list
        list of Bond objects in the system
    box : np.ndarray
        box dimension of the system
    int_fact : IntegratorFactory
        integrator factory used to get the integrators used in the system
    integrator : Integrator
        integrator used to step the system through time
    central_potential : Potential
        central potential applied to the system every step
    bc : BoundaryCondition
        boundary condition applied to the system edges
    dim : int
        dimensions of system
    dof : int
        degrees of freedom of the system

    Methods
    -------
    registerObserver(Observer)
        register an observers to be notified of state changes
    removeObserver(Observer)
        remove observer from observer list
    notifyObserver(steps)
        notify Observers giving the number of steps the system is on
    get_integrator(integrator_name, **kwargs)
        use the integrator facotry to build an system integrator
    get_thermostat(thermostat_name, T, **kwargs)
        use the integrator's thermostat factory tob build a thermostat for
        the integrator
    run(steps)
        run the system for a certain number of steps
    add_particle(particle)
        add particle to the system
    add_bond(bond)
        add bond to the system
    apply_bc(coords)
        apply the boundary conditions of the system to a set of coordinates
    get_velocities()
        assemble particle velocities of the particles into a numpy array
    set_velocities(vels, indices)
        set the velocities of particles using a numpy array
    get_coordinates()
        assemble particle coordinates into a numpy array
    set_coordinates(coords, indices)
        set particle coordinates using a numpy array
    get_energy()
        calculate the system energy using the integrator
    get_masses()
        get the system particle masses
    set_masses(masses, indices)
        set the system particle masses with a numpy array
    get_forces()
        get the per particle system forces in a numpy array
    set_forces(forces, indices)
        set the forces per particle using a numpy array
    """
    def __init__(self, box = np.array([1, 1]), dim = 2):
        super().__init__()
        self.particles = []
        self.bonds = []
        self.box = box
        self.int_fact = integrators.IntegratorFactory(self)
        self.integrator = None
        self.central_potential = None
        self.bc = boundary_conditions.NoBoundaryConditions(self)
        self.dim = dim
        self.dof = None

    # Observer functions

    def registerObserver(self, Observer):
        self.observers.append(Observer)

    def removeObserver(self, Observer):
        self.observers.remove(Observer)

    def notifyObservers(self, steps):
        for obs in self.observers:
            obs.update(steps)

    # Get type of integrator
    def get_integrator(self, integrator_name, **kwargs):
        self.integrator = self.int_fact.get_integrator(integrator_name, **kwargs)

    def get_thermostat(self, thermostat_name, T, **kwargs):
        self.integrator.get_thermostat(thermostat_name, T, **kwargs)

    # Run Simulation
    def run(self, steps):
        for step in range(steps):
            self.integrator.integrate()
            self.notifyObservers(step + 1)
        

    def add_particle(self, particle):
        if particle.force is None:
            particle.force = 0
        if particle.vel is None:
            particle.vel = 0
        self.particles.append(particle)


    # def __init__(self, potential, particle_1, particle_2, particle_interactions = False, bc = None):

    def add_bond(self, potential, particle_i, particle_j):
        particle_1 = self.particles[particle_i]
        particle_2 = self.particles[particle_j]
        self.particles.insert(0, self.particles.pop(particle_j))
        self.particles.insert(0, self.particles.pop(self.particles.index(particle_1)))
        self.bonds.append(Bond(potential, particle_1, particle_2, bc = self.bc))



    # Boundary Conditions
    def apply_bc(self, coords):
        return self.bc(coords)

    # Get/Set system states
    def get_velocities(self):
        vels = np.zeros((len(self.particles), self.dim))
        for i in range(len(self.particles)):
            vels[i, :] = self.particles[i].vel
        return vels

    def set_velocities(self, vels, indices = []):
        if  len(indices) == 0:
            indices = list(range(len(self.particles)))
        
        for i in range(len(indices)):
            # print("Updating Particle", i, "velocity")
            # print(vels[i,:])
            self.particles[indices[i]].vel = vels[i, :]

    def get_coordinates(self):
        coords = np.zeros((len(self.particles), self.dim))
        for i in range(len(self.particles)):
            coords[i ,:] = self.particles[i].loc
        return coords

    def set_coordinates(self, coords, indices = []):
        if  len(indices) == 0:
            indices = range(len(self.particles))
        
        for i in indices:
            self.particles[i].loc = coords[i, :]

    def set_coordinates_tf(self, coords, indices =[]):
        if  len(indices) == 0:
            indices = range(len(self.particles))
        
        for i in indices:
            self.particles[i].loc = coords[i, :]

    def get_forces(self):
        forces = np.zeros((len(self.particles), self.dim))
        for i in range(len(self.particles)):
            forces[i ,:] = self.particles[i].force
            # print(forces)
        return forces

    def set_forces(self, forces, indices = []):
        if  len(indices) == 0:
            indices = range(len(self.particles))
        
        for i in indices:
            # print(forces[i, :])
            self.particles[i].force = forces[i, :]

    def get_masses(self):
        masses = np.zeros(len(self.particles))
        for i in range(len(self.particles)):
            masses[i] = self.particles[i].mass
        return masses

    def set_masses(self, masses, indices = []):
        if  len(indices) == 0:
            indices = range(len(self.particles))
        
        for i in indices:
            self.particles[i].masses = masses[i]

    def get_energy(self):
        H, U, K = self.integrator.calculate_energy()
        return(H, U, K)

            
        

