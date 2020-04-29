import sys
import numpy as np
import pandas as pd
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


def get_config(sim):
    d = {
        "Double Well": "../notebooks/configs/double_well_config_1.yml",
        "Mueller Potential": "../notebooks/configs/muller_well_config_1.yml",
        "Bistable Dimer in LJ Fluid": "../notebooks/configs/dimer_sim_config_md_1.yml"
    }
    return d[sim]
    
def get_rc(sim):
    d = {
        "Bistable Dimer in LJ Fluid": lambda x: np.linalg.norm(np.array(x[0:2]) - np.array(x[2:4])),
        "Double Well": lambda x: x[0],
        "Mueller Potential": lambda x: np.dot(x, np.array([1, -1]))/np.dot(np.array([1,-1]),np.array([1,-1]))
    }
    return d[sim]
    
def get_limits(sim):
    d = {
        "Double Well": [[-3.8,3.4], [-4.5,4.5]],
        "Mueller Potential": [[-1.5, 1.1], [-0.5, 2.]]
    }
    return d[sim]

@st.cache(suppress_st_warning=True, hash_funcs={RealNVP: hash})
def initialize_model(path):
    loss = getLoss().ml_loss()
    opt = getOpt().rmsprop(1e-4)
    m = RealNVP(loss, opt)
    if path is not "":
        try:
            m.load(path)
            st.sidebar.text("Accepted model type")
        except tf.errors.NotFoundError:
            st.sidebar.text("Not a valid model path")
    return m


@st.cache(suppress_st_warning=True, hash_funcs={RealNVP: hash})
def generate_image(num_samples, model):
    target = model.forward_sample(num_samples)
    gauss = model.backward_sample(target)

    fig1 = plt.figure(figsize=(10,10), dpi=150)
    plt.scatter(target.numpy().T[0], target.numpy().T[1])
    im1 = np.array(fig2img(fig1))
    plt.close()

    fig2 = plt.figure(figsize=(10,10), dpi=150)
    plt.scatter(gauss.numpy().T[0], gauss.numpy().T[1])
    im2 = np.array(fig2img(fig2))
    plt.close()
    return im1, im2


@st.cache(suppress_st_warning=True, hash_funcs={RealNVP: hash})
def plot_samples(model, n, sim):
    fig = plt.figure(figsize=(12, 8), dpi=125)
    targets = model.forward_sample(n).numpy().T
    if not sim == "Double Moon":
        simulation = SimulationData(get_config(sim))
        limits = get_limits(sim)
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
def plot_free_energy(model, n, sim):
    rc_samples = []
    weights = []

    rc_func = get_rc(sim)
    config = get_config(sim)
    simulation = SimulationData(config)
    
    targets = model.forward_sample(n).numpy()
    probs = model.flow.log_prob(targets, events_ndims=targets.shape[1]).numpy()
    if(len(probs.shape) > 1):
        probs = probs.diagonal()
    
    for (t, lp) in list(zip(targets, probs)):
        rc_samples.append(rc_func(t))
        if sim == "Bistable Dimer in LJ Fluid":
            t = t.reshape((36, 2))
        else:
            t = t.reshape((1, 2))
        weights.append(np.exp(-simulation.getEnergy(t) + lp))
    
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




def main():
    
    num_samples = st.sidebar.slider("Number of samples to run", min_value=500, max_value=25000, value=5000)
    sim = st.sidebar.selectbox("Which data set was the model trained on?", ["Double Moon", "Double Well", "Mueller Potential", "Bistable Dimer in LJ Fluid"])
    if sim == "Double Moon":
        plot_choice = "Forward Samples"
    elif sim == "Bistable Dimer in LJ Fluid":
        plot_choice = st.sidebar.selectbox("Plots", ["Free Energy", "Dimer Positions"])
    else:
        plot_choice = st.sidebar.selectbox("Plots", ["Free Energy", "Forward Samples"])
    model_file = st.sidebar.text_input("Path to a saved checkpoint", value="")
    model = initialize_model(model_file)
    
    if plot_choice == "Forward Samples":
        plot_samples(model, num_samples, sim)
    if plot_choice == "Free Energy":
        plot_free_energy(model, num_samples, sim)
    
        




if __name__ == "__main__":
    main()