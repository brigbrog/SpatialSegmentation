import numpy as np
import pandas as pd
#import cv2 
#from scipy.ndimage import gaussian_filter
import seaborn as sb
import tifffile as tiff
from multiprocessing import Pool, cpu_count
import sys
import progressbar
import requests 
import time
import os
from tqdm import tqdm

import gc
from joblib import Parallel, delayed

# git ignore for input output dir 

class Comparator:
    def __init__(self, 
                 origin_csv_fname=None,
                 mask_maker_pq=None,
                 ):
        self.origin_csv_fname = origin_csv_fname
        self.mask_maker_pq = mask_maker_pq