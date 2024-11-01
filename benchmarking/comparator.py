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
# need to find how to get indicator lists

class Comparator:
    def __init__(self, 
                 origin_csv_fname: str = None,
                 mask_maker_pqfname: str = None,
                 pos_indicators: str = None,
                 neg_indicators: str = None,
                 rep_perc: float = 1.0
                 ):
        self.origin_csv_fname = origin_csv_fname
        self.mask_maker_pq = mask_maker_pqfname
        self.pos_indicators = pos_indicators # convert from fname (strings in txt) to array of str (names)
        self.neg_indicators = neg_indicators # convert from fname (strings in txt) to array of str (names)
        self.metadata = None # read parquet -> DataFrame
        self.contours = None # read json -> DataFrame
        self.markers = None # read json -> DataFrame
        self.rep_perc = rep_perc
        self.testidx = self.set_testidx()
        

    def import_metadata(self):
        pass

    def import_contours(self):
        pass 

    def import_markers(self):
        pass

    def set_testidx(self):
        pass

    def thin(self, 
             set: pd.DataFrame = None
             ):
        pass

    def run_comparison(self, 
                       set: pd.DataFrame = None
                       ):
        pass

    def compare_engine(self):
        pass

    def export_compare_data(self, 
                            df: pd.DataFrame = None,
                            ):
        pass


