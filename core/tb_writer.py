import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt

from PIL import Image
from tools import fig2img
from matplotlib.figure import Figure

class ObserverInterface:
    def __init__(self):
        pass
    def update(self):
        raise NotImplementedError


class tensorboard_writer(ObserverInterface):
    def __init__(self, writer):
        """
         Observer that parses training data onto the tensorboard

         PARAMETER:
         * writer: a tensorboard summary writer object initialized by RealNVP
        """
        super(tensorboard_writer, self).__init__()
        self.writer = writer


    def update(self, state):
        for k,v in state.items():
            if(type(v) == tuple):   
                if(isinstance(v[0], (tf.Tensor))):
                    self.process_tensors(k,v)            
                elif(isinstance(v[0], (Figure))):
                    self.process_images(k, v[0], v[1])
                elif(isinstance(v[0], (int, float, ))):
                    self._update_scalers(k, v[0], v[1])
        

    def process_tensors(self, k,v):
        if(isinstance(v[0].numpy(), (np.floating))):
            self._update_scalers(k, v[0], v[1])


    def _update_scalers(self,tag, value, step):
        with self.writer.as_default():
            tf.summary.scalar(tag, value, step=step)


    def process_images(self, tag, fig, step):
        im = np.array(fig2img(fig))
        im = im.reshape((-1, im.shape[0], im.shape[1], im.shape[2]))
        with self.writer.as_default():
            tf.summary.image(tag, im, step=step)

    
    