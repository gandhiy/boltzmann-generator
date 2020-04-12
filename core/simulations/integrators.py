# integrator objects

from abc import ABC, abstractmethod
import numpy as np
import itertools

class Integrator(ABC):
    def __init__(self, system, dt):
        self.system = system
        self.dt = dt

    @abstractmethod
    def integrate(self):
        pass

class IntegratorFactory():
    def __init__(self, system):
        self.system = system
        self.integrators = { 
                            "verlet" : VerletIntegrator,
                            "verlet_neighbors" : VerletIntegratorNeighbors,
                            "verlet_cell_list" : VerletIntegratorCellList
                            }

    def get_integrator(self, integrator_name, dt, **kwargs):
        if integrator_name in self.integrators.keys():
            return(self.integrators[integrator_name](self.system, dt,**kwargs))
        else:
            raise NotImplementedError(integrator_name + " is not a valid integrator")

class VerletIntegrator(Integrator):
    def __init__(self, system, dt):
        super().__init__(system, dt)

    def integrate(self):
        r = self.system.get_coordinates()
        v = self.system.get_velocities()
        f = self.system.get_forces()

        v_half = v + np.multiply(f.transpose(), self.dt / ( 2 * self.system.get_masses() )).transpose()
        r += v_half * self.dt
        r = self.system.apply_pbc(r)
        self.system.set_coordinates(r)
        f = self.calculate_forces()
        v = v_half + np.multiply(f.transpose(), self.dt / ( 2 * self.system.get_masses() )).transpose()
        
        self.system.set_velocities(v)
        self.system.set_forces(f)

    def calculate_forces(self):
        indices = list(range(len(self.system.particles)))
        forces = np.zeros((len(self.system.particles), self.system.dim))
        # print(forces)
        for pair in itertools.combinations(indices, 2):
            i, j = pair[0], pair[1]
            if i == j:
                continue
            r_ij = np.array(self.system.apply_pbc(self.system.particles[i].loc - self.system.particles[j].loc))
            r_ji = np.array(self.system.apply_pbc(self.system.particles[j].loc - self.system.particles[i].loc))
            forces[i, :] += self.system.particles[j].potential.derivative(r_ij)
            forces[j, :] += self.system.particles[i].potential.derivative(r_ji)
        return forces

    def calculate_energy(self):
        U = 0
        vs = self.system.get_velocities()
        K = 0.5 * np.sum(self.system.get_masses() *  np.array([np.dot(vs[i,:], vs[i,:]) for i in range(vs.shape[0])]))
        for pair in itertools.combinations(range(len(self.system.particles)), 2):
            i, j = pair[0], pair[1]
            r_ij = np.array(self.system.apply_pbc(self.system.particles[i].loc - self.system.particles[j].loc))
            r_ji = - r_ij
            u_ij = self.system.particles[j].potential(r_ij)
            u_ji = self.system.particles[i].potential(r_ji)
            U += (u_ij + u_ji)/2
        H = U + K
        return H, U, K

    def calculate_pressure(self):
        w = 0
        indices = list(range(len(self.system.particles)))
        for pair in itertools.combinations(indices, 2):
            i, j = pair[0], pair[1]
            if i == j:
                continue
            r_ij = np.array(self.system.apply_pbc(self.system.particles[i].loc - self.system.particles[j].loc))
            r_ji = np.array(self.system.apply_pbc(self.system.particles[j].loc - self.system.particles[i].loc))
            f_ij = self.system.particles[j].potential.derivative(r_ij)
            f_ji = self.system.particles[i].potential.derivative(r_ji)
            w += (1 / self.system.dim) * ( np.dot(f_ij, r_ij) + np.dot(f_ji,r_ji))
        T_i = self.calculate_temperature()
        P = w / self.system.box.prod() + len(self.system.particles) * T_i 
        return(P)

    def calculate_temperature(self):
        vs = self.system.get_velocities()
        K = 0.5 * np.sum(self.system.get_masses() *  np.array([np.dot(vs[i,:], vs[i,:]) for i in range(vs.shape[0])]))
        T_i = 2 * K / (self.system.dim * len(self.system.particles) - self.system.dim)
        return(T_i)


class VerletIntegratorNeighbors(VerletIntegrator):
    def __init__(self, system, dt, r_nl):
        super().__init__(system, dt)
        self.r_nl = r_nl
        self.init_particle_history()
        self.update_neighbors()

    
    def init_particle_history(self):
        for particle in self.system.particles:
            particle.prev_loc = particle.loc
            particle.delta_r = 0
            
    def integrate(self):
        r = self.system.get_coordinates()
        v = self.system.get_velocities()
        f = self.system.get_forces()

        v_half = v + np.multiply(f.transpose(), self.dt / ( 2 * self.system.get_masses() )).transpose()
        r += v_half * self.dt
        r = self.system.apply_pbc(r)
        self.system.set_coordinates(r)
        self.check_delta_r()
        f = self.calculate_forces()
        v = v_half + np.multiply(f.transpose(), self.dt / ( 2 * self.system.get_masses() )).transpose()
        
        self.system.set_velocities(v)
        self.system.set_forces(f)

    def check_delta_r(self):
        for particle in self.system.particles:
            r = self.system.apply_pbc(particle.loc - particle.prev_loc)
            delta_r_sq = np.dot(r,r)
            del_sq = (self.r_nl - particle.potential.r_c)**2
            if delta_r_sq > del_sq/4:
                self.update_neighbors()
                break


    def update_neighbors(self):
        neighbor_table =  np.zeros([len(self.system.particles)]*2)
        for particle in self.system.particles:
            particle.neighbors = []
            particle.prev_loc = particle.loc
        for pair in itertools.combinations(range(len(self.system.particles)), 2):
            i, j = pair[0], pair[1]
            r = self.system.apply_pbc(self.system.particles[i].loc - self.system.particles[j].loc)
            r_sq = np.dot(r, r)
            if  r_sq < self.r_nl**2:
                neighbor_table[i,j] += 1
                neighbor_table[j,i] += 1
                self.system.particles[i].neighbors.append(self.system.particles[j])
                self.system.particles[j].neighbors.append(self.system.particles[i])
        
    def calculate_forces(self):
        indices = list(range(len(self.system.particles)))
        forces = np.zeros((len(self.system.particles), self.system.dim))
        for i in indices:
            for neighbor in self.system.particles[i].neighbors:
                r_ij = np.array(self.system.apply_pbc(self.system.particles[i].loc - neighbor.loc))
                # print(neighbor.potential.derivative(r_ij))
                forces[i, :] += neighbor.potential.derivative(r_ij)
        return forces

    def calculate_energy(self):
        U = 0
        vs = self.system.get_velocities()
        K = 0.5 * np.sum(self.system.get_masses() *  np.array([np.dot(vs[i,:], vs[i,:]) for i in range(vs.shape[0])]))
        for i in range(len(self.system.particles)):
            for neighbor in self.system.particles[i].neighbors:
                r_ij = np.array(self.system.apply_pbc(self.system.particles[i].loc - neighbor.loc))
                U += neighbor.potential(r_ij)
        H = U + K
        return H, U, K


    def calculate_pressure(self):
        w = 0
        indices = list(range(len(self.system.particles)))

        for i in indices:
            for neighbor in self.system.particles[i].neighbors:
                r_ij = np.array(self.system.apply_pbc(self.system.particles[i].loc - neighbor.loc))
                f_ij = neighbor.potential.derivative(r_ij)
                w += (1 / self.system.dim) * np.dot(f_ij, r_ij)
        T_i = self.calculate_temperature()
        P = w / self.system.box.prod() + len(self.system.particles) * T_i 
        return(P)

class VerletIntegratorCellList(VerletIntegrator):
    def __init__(self, system, dt, r_cell):
        super().__init__(system, dt)
        self.r_cell = self.system.box / np.floor(self.system.box / r_cell)
        print("Adjusted R_cell:", self.r_cell)
        self.init_cells_lists()
        self.neighbor_indexes = list(itertools.product([-1, 0, 1], repeat = len(self.system.box)))
        self.cell_indexes = list(itertools.product(*[range(int(dim)) for dim in self.cells.shape]))


    def init_cells_lists(self):
        cell_dims = np.round(self.system.box / self.r_cell).astype(int)
        self.cells = np.empty(cell_dims.astype(int), dtype=object)
        for pos in itertools.product(*[range(int(dim)) for dim in cell_dims]):
            self.cells[pos] = []

        for i in range(len(self.system.particles)):
            cell_i = tuple(np.floor((self.system.particles[i].loc + self.system.box / 2) / self.r_cell).astype(int))
            self.system.particles[i].cell = cell_i
            self.system.particles[i].index = i
            self.cells[cell_i].append(self.system.particles[i])

    def calculate_forces(self):
        indices = list(range(len(self.system.particles)))
        forces = np.zeros((len(self.system.particles), self.system.dim))
        for i in indices:
            neighbors = self.get_cell_neighbors(self.system.particles[i].cell)
            for neighbor in neighbors:
                if neighbor == self.system.particles[i]:
                    continue
                r_ij = np.array(self.system.apply_pbc(self.system.particles[i].loc - neighbor.loc))
                forces[i, :] += neighbor.potential.derivative(r_ij)
        return forces

    def calculate_energy(self):
        U = 0
        vs = self.system.get_velocities()
        K = 0.5 * np.sum(self.system.get_masses() *  np.array([np.dot(vs[i,:], vs[i,:]) for i in range(vs.shape[0])]))
        for i in range(len(self.system.particles)):
            neighbors = self.get_cell_neighbors(self.system.particles[i].cell)
            for neighbor in neighbors:
                if neighbor == self.system.particles[i]:
                    continue
                r_ij = np.array(self.system.apply_pbc(self.system.particles[i].loc - neighbor.loc))
                U += neighbor.potential(r_ij)
        H = U/2 + K
        return H, U, K


    def calculate_pressure(self):
        w = 0
        indices = list(range(len(self.system.particles)))
        for i in indices:
            neighbors = self.get_cell_neighbors(self.system.particles[i].cell)
            for neighbor in neighbors:
                if neighbor == self.system.particles[i]:
                    continue
                r_ij = np.array(self.system.apply_pbc(self.system.particles[i].loc - neighbor.loc))
                f_ij = neighbor.potential.derivative(r_ij)
                w += (1 / self.system.dim) * np.dot(f_ij, r_ij)
        T_i = self.calculate_temperature()
        P = w / self.system.box.prod() + len(self.system.particles) * T_i 
        return(P)
    
    def get_cell_neighbors(self, index):
        cell_neigh_particles = []
        # print("Center:", index)
        for cell_indexes in self.neighbor_indexes:
            cell_indexes = tuple((np.array(index) + np.array(cell_indexes)) % np.array(self.cells.shape))
            # if cell_indexes == index:
            #    continue
            cell_neigh_particles.extend(self.cells[cell_indexes])
        return cell_neigh_particles

    def update_cells(self):
        for particle in self.system.particles:
            # print(particle.loc)
            cell_i = tuple(np.floor((particle.loc + self.system.box / 2) / self.r_cell).astype(int))
            # print(particle.loc)
            if particle.cell == cell_i:
                continue
            else:
                # print("Moving particle from cell", particle.cell, "to cell", cell_i)
                self.cells[particle.cell].remove(particle)
                self.cells[cell_i].append(particle)
                particle.cell = cell_i


    def integrate(self):
        r = self.system.get_coordinates()
        v = self.system.get_velocities()
        f = self.system.get_forces()

        v_half = v + np.multiply(f.transpose(), self.dt / ( 2 * self.system.get_masses() )).transpose()
        r += v_half * self.dt
        r = self.system.apply_pbc(r)
        self.system.set_coordinates(r)
        self.update_cells()
        f = self.calculate_forces()
        v = v_half + np.multiply(f.transpose(), self.dt / ( 2 * self.system.get_masses() )).transpose()
        
        self.system.set_velocities(v)
        self.system.set_forces(f)