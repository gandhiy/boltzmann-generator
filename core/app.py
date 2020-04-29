import sys
import numpy as np
import streamlit as st
import tensorflow as tf
import matplotlib.pyplot as plt

sys.path.append("simulations/")

from tools import fig2img
from losses import getLoss
from optimizers import getOpt
from network_base import RealNVP
from simulations.visuals import *
from simulations.simulations import SimulationData

plt.style.use("fivethirtyeight")







# @st.cache(suppress_st_warning=True, hash_funcs={RealNVP: hash})
# def control_plots(plot, model, n, sim):



    
class app:
    def __init__(self):
        pass


    def get_config(self, sim):
        d = {
            "Double Well": "../notebooks/configs/double_well_config_1.yml",
            "Mueller Potential": "../notebooks/configs/muller_well_config_1.yml",
            "Bistable Dimer in LJ Fluid": "../notebooks/configs/dimer_sim_config_md_1.yml"
        }
        return d[sim]
        


    def get_rc(self, sim):
        d = {
            "Bistable Dimer in LJ Fluid": lambda x: np.linalg.norm(np.array(x[0:2]) - np.array(x[2:4])),
            "Double Well": lambda x: x[0],
            "Mueller Potential": lambda x: np.dot(x, np.array([1, -1]))/np.dot(np.array([1,-1]),np.array([1,-1]))
        }
        return d[sim]
        


    def get_limits(self, sim):
        d = {
            "Double Well": [[-3.8,3.4], [-4.5,4.5]],
            "Mueller Potential": [[-1.5, 1.1], [-0.5, 2.]]
        }
        return d[sim]




    @st.cache(suppress_st_warning=True, hash_funcs={RealNVP: hash})
    def initialize_model(self, path, sim):
        loss = getLoss().ml_loss()
        opt = getOpt().rmsprop(1e-4)
        if sim == 'Bistable Dimer in LJ Fluid':
            m = RealNVP(loss, opt, in_shape=[72], loc=[0.]*72, scale=[1.]*72, model_name=None)
        else:
            m = RealNVP(loss, opt)
        if path is not "":
            try:
                m.load(path)
                st.sidebar.text("Accepted model type")
            except tf.errors.NotFoundError:
                st.sidebar.text("Not a valid model path")
        return m



    
    @st.cache(suppress_st_warning=True, hash_funcs={RealNVP: hash})
    def plot_samples(self, model, n, sim):
        fig = plt.figure(figsize=(12, 8), dpi=125)
        targets = model.forward_sample(n).numpy().T
        if not sim == "Double Moon":
            simulation = SimulationData(self.get_config(sim))
            limits = self.get_limits(sim)
            plot_2D_potential(simulation.simulation.central_potential, xlim=limits[0], ylim=limits[1], cmap='jet')
            plt.xlim(limits[0])
            plt.ylim(limits[1])
        plt.title(f"Samples from the {sim} trained model")
        plt.xlabel(r"$x_1$")
        plt.ylabel(r"$x_2$")
        plt.scatter(targets[0], targets[1], c='black', s=1.15, cmap='jet')
        plt.close()
        im = np.array(fig2img(fig))
        st.image(im, caption='Forward Samples', use_column_width=True)



    @st.cache(suppress_st_warning=True, hash_funcs={RealNVP: hash})
    def plot_free_energy(self, model, n, sim):
        bar = st.progress(0)
        inc = 1.0/n
        rc_samples = []
        weights = []

        rc_func = self.get_rc(sim)
        config = self.get_config(sim)
        simulation = SimulationData(config)
        
        targets = model.forward_sample(n).numpy()
        log_prob = model.flow.log_prob(targets, events_ndims=targets.shape[1]).numpy()
        if(len(log_prob.shape) > 1):
            log_prob = log_prob.diagonal()
        
        for i,(t, lp) in enumerate(list(zip(targets, log_prob))):
            rc_samples.append(rc_func(t))
            if sim == "Bistable Dimer in LJ Fluid":
                t = t.reshape((36, 2))
            else:
                t = t.reshape((1, 2))
            weights.append(np.exp(-simulation.getEnergy(t) + lp))
            bar.progress((i*1.0)/n + inc)
        
        counts, bins = np.histogram(rc_samples, weights=weights, bins=200)
        probs = (counts / np.sum(counts)) + 1e-9
        bin_centers = (bins[:-1] + bins[1:])/2.0
        FE = -np.log(probs)

        fig = plt.figure(figsize= (12, 8), dpi = 125)
        plt.xlabel("Reaction Coordinate")
        plt.ylabel("Free Energy")
        plt.plot(bin_centers[FE < -np.log(1e-9)], FE[FE < -np.log(1e-9)])
        plt.close()
        im = np.array(fig2img(fig))
        st.image(im, "Free Energy Diagram", use_column_width=True)





    @st.cache(suppress_st_warning=True, hash_funcs={RealNVP: hash})
    def plot_dimer_positions(self, model, n, sim, alpha):
        targets = model.forward_sample(n).numpy().reshape((-1, 36, 2))
        ap = np.mean(targets, axis=0)
        fig = plt.figure(figsize=(12, 8), dpi=125)
        plt.xlabel(" x positions ")
        plt.ylabel(" y positions ")
        plt.plot(ap[0:2, 0], ap[0:2, 1], c='red', linewidth=8/3, zorder=10)
        plt.plot(ap[0:2, 0], ap[0:2, 1], 'o', c='red', markersize=8, zorder=9)
        plt.plot(ap[2:, 0], ap[2:, 1], 'o', c='blue', markersize=4, zorder=8)

        bar = st.progress(0)
        inc = 1.0/n
        for i in range(targets.shape[1]):
            if i < 2:
                plt.plot(targets[:, i, 0], targets[:, i, 1], 'o', c='red', alpha=min(alpha*2, 1), markersize=4, zorder=4)
            else:
                plt.plot(targets[:, i, 0], targets[:, i, 1], 'o', c='blue', alpha=alpha, markersize=2, zorder=0)
            bar.progress((i * 1.0)/targets.shape[1] + (1.0/targets.shape[1]))

        im = np.array(fig2img(fig))
        st.image(im, "Dimer Positions", use_column_width=True)

    
    def main(self):
        num_samples = st.sidebar.slider("Number of samples to run", min_value=500, max_value=25000, value=5000)
        sim = st.sidebar.selectbox("Which data set was the model trained on?", ["Double Moon", "Double Well", "Mueller Potential", "Bistable Dimer in LJ Fluid"])
        if sim == "Double Moon":
            plot_choice = "Forward Samples"
        elif sim == "Bistable Dimer in LJ Fluid":
            plot_choice = st.sidebar.selectbox("Plots", ["Free Energy", "Dimer Positions"])
        else:
            plot_choice = st.sidebar.selectbox("Plots", ["Free Energy", "Forward Samples"])
        model_file = st.sidebar.text_input("Path to a saved checkpoint", value="")
        model = self.initialize_model(model_file, sim)
        
        # control_plots(plot_choice, model, num_samples, sim)
        if plot_choice == "Forward Samples":
            self.plot_samples(model, num_samples, sim)
        if plot_choice == "Free Energy":
            self.plot_free_energy(model, num_samples, sim)
        if plot_choice == "Dimer Positions":
            alpha = st.sidebar.slider("alpha", min_value = 0.01, max_value=1.0,value=0.15, step=0.001)
            self.plot_dimer_positions(model, num_samples, sim, alpha)
            




if __name__ == "__main__":
    application = app()
    application.main()