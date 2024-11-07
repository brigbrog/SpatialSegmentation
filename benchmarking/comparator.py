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
import fastparquet
from tqdm import tqdm

import gc
from joblib import Parallel, delayed

# git ignore for input output dir
# need to find how to get indicator lists

class Comparator:
    def __init__(self, 
                 origin_csv_fname: str = None,
                 pqfname: str = None,
                 pos_indicators: str = None,
                 neg_indicators: str = None,
                 rep_perc: float = 1.0
                 ):
        self.origin_csv_fname = origin_csv_fname
        self.pqfname = pqfname
        self.pos_indicators = pos_indicators # convert from fname (strings in txt) to array of str (names)
        self.neg_indicators = neg_indicators # convert from fname (strings in txt) to array of str (names)
        self.rep_perc = rep_perc
        # use tqdm to display importation tasks #
        self.testidx = self.set_testidx() 
        self.metadata = self.import_metadata() # read parquet -> DataFrame 
        self.contours = self.import_contours() # read json -> DataFrame
        self.markers = self.import_markers() # read json -> DataFrame
    

    def import_metadata(self):
        metadata = pd.read_parquet(self.pqfname, engine='fastparquet')
        return metadata

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
                       mask_df: pd.DataFrame = None
                       ):
        pass

    def compare_engine(self):
        pass

    def export_compare_data(self, 
                            out_df: pd.DataFrame = None,
                            ):
        pass


def test_import_metadata(pqfname):
    metadata = pd.read_parquet(pqfname, engine='fastparquet')
    return metadata


if __name__ == '__main__':
    metadata = test_import_metadata(
        '/Users/brianbrogan/Desktop/KI24/SpatialSegmentation/mask_maker_output/variable_segmentation_metadata.parquet'
    )
    print(metadata.head(10))
