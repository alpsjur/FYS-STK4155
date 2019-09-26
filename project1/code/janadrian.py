from sklearn.utils import shuffle
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
from matplotlib import cm
from matplotlib.ticker import LinearLocator, FormatStrFormatter
import numpy as np
from random import random, seed
from imageio import imread

import projectfunctions as pf


oslo_data = imread('../data/test_data_oslo.tif')

print(oslo_data.shape)
