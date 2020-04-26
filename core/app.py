import numpy as np
import pandas as pd
import streamlit as st
import tensorflow as tf
import matplotlib.pyplot as plt

from tools import fig2img
from losses import getLoss
from optimizers import getOpt
from network_base import RealNVP

plt.style.use("fivethirtyeight")


def main():
    num_samples = st.sidebar.slider("Number of samples to run", min_value=500, max_value=25000, value=5000)
    
    plot_choice = st.sidebar.selectbox("Which diagram plot?", ["Free Energy", "Samples"])
    model_file = st.sidebar.text_input("Path to a saved checkpoint", value="")
    model = initialize_model(model_file)
    
    
    if plot_choice == 'Samples':
        forward, backward = generate_image(num_samples, model)
        st.image(forward, caption="Forward Sampling from RealNVP", use_column_width=True)
        st.image(backward, caption="Backward Gaussian from above Target", use_column_width=True)



@st.cache(suppress_st_warning=True, hash_funcs={RealNVP: hash})
def initialize_model(path):
    loss = getLoss().maximum_likelihood_loss()
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


if __name__ == "__main__":
    main()