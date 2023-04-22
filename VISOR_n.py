# Visor for n objects

import numpy as np
import cv2
from ImageReconstruction.neural_nexus_clipseg import getMask



class VISOR_n(object):
    def __init__(self, img, objects, directions):
        assert(type(img) == np.ndarray)
        assert(type(object) == list)
        assert(type(directions) == list)
        self._img = img
        self._objects = objects
        self._directions = directions
    
    # get masks for all the objects
    def get_mask(self):
        