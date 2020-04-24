import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt

from PIL import Image
from tools import fig2img

class ObserverInterface:
    def __init__(self):
        pass
    def update(self):
        raise NotImplementedError


class tensorboard_writer(ObserverInterface):
    def __init__(self, writer):
        super(tensorboard_writer, self).__init__()
        self.writer = writer


    def update(self, state):
        for k,v in state.items():
            if(type(v) == tuple):   
                if(isinstance(v[0], (tf.Tensor))):
                    self.process_tensors(k,v)            
                

    def process_tensors(self, k,v):
        if(isinstance(v[0].numpy(), (np.floating))):
            self._update_scalers(k, v[0], v[1])
        elif(isinstance(v[0].numpy(), (np.ndarray))):
            self._update_images(k, v[0], v[1])

    def _update_scalers(self,tag, value, step):
        with self.writer.as_default():
            tf.summary.scalar(tag, value, step=step)


    def _update_images(self, tag, value, step):
        fig = plt.figure(figsize=(10,10), dpi=100)
        plt.scatter(value.numpy().T[0], value.numpy().T[1])
        im = np.array(fig2img(fig))
        plt.close()
        im = im.reshape((-1, im.shape[0], im.shape[1], im.shape[2]))
        with self.writer.as_default():
            tf.summary.image(tag, im, step=step)

    
    