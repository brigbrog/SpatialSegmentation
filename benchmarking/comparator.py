import numpy as np
import pandas as pd
import seaborn as sb
import tifffile as tiff
import matplotlib.pyplot as plt
import cv2
from multiprocessing import Pool, cpu_count
import sys
import progressbar
import requests 
import time
import os
import fastparquet
from tqdm import tqdm
import json
from scipy.spatial import KDTree

import gc
from joblib import Parallel, delayed

# git ignore for input output dir DONE
# need to find how to get indicator lists DONE
# FIX MARKERS and CONTOURS importers, they are fucked up
    #contours DONE
    #markers DONE
# Make testidx and thinner before pos/neg masks
# Use the DataManager
# create pos/neg masks for quick comparison to origin
    # Only have cell specific true pos false neg, false pos and true neg are rates bc no cell assignment
    # create numpy mask for each contour and pull positive coords or find better way...
    # use indicatorFinder object (maybe attribute ?) DONE
    # remember representative percentage

class Indicator:
        #need to add some kind of dictionary support for mask making
    def __init__(self,
                 origin_csv: pd.DataFrame,
                 annotation: pd.DataFrame,
                 indicator_minimum: int = 30000,
                 positive_id: int = 0,
                 negative_ids: int|list = [4,5,11,13]
                 ):
        self.origin_csv = origin_csv
        self.annotation = annotation
        self.indicator_minimum = indicator_minimum
        self.positive_id = positive_id
        self.negative_ids = negative_ids
        self.positive_indicators = self.find_pos_indicators()
        self.negative_indicators = self.find_neg_indicators()
        self.indicators = self.create_full_indicator_df()

    def find_gene_set(self,
                      id: int = None):
        assert id is not None, \
            "Cluster ID must be provided to pull indicators."
        clust = self.annotation.loc[self.annotation['cluster']==id, :]
        clust_genes = set(np.unique(clust['features']))
        return clust_genes
    
    def create_cluster_df(self,
                          gene_set: pd.Series = None
                          ):
        cluster_df = self.origin_csv[self.origin_csv['geneID'].isin(gene_set)]
        return cluster_df
        
    def filter_indicators(self, 
                          cluster_df: pd.DataFrame = None,
                          indicator_type: str = 'min_count',
                          n: int = 5
                          ):
        assert indicator_type == 'min_count' or indicator_type == 'spec',\
            "indicator_type must either be 'min_count' or 'spec'."
        indicators = cluster_df['geneID'].value_counts()
        if indicator_type == 'min_count':
            indicators = indicators.loc[indicators>= self.indicator_minimum]
        elif indicator_type == 'spec':
            indicators = indicators.iloc[:n]
        return indicators
    
    def find_pos_indicators(self):
        with tqdm(total=3, desc="Finding positive indicators", unit="step") as pbar:
            pos_genes = self.find_gene_set(self.positive_id)
            pbar.update(1)
            pos_cluster_df = self.create_cluster_df(pos_genes)
            pbar.update(1)
            pos_indicators = self.filter_indicators(pos_cluster_df)
            pbar.update(1)
        return pos_indicators
    
    def find_neg_indicators(self):
        neg_indicators = pd.DataFrame(columns=['geneID', 'count', 'cluster', 'type'])
        with tqdm(total=len(self.negative_ids), desc="Finding negative indicators", unit="cluster") as pbar:
            for clust_id in self.negative_ids:
                clust_genes = self.find_gene_set(clust_id)
                cluster_df = self.create_cluster_df(clust_genes)
                cluster_indicators = self.filter_indicators(cluster_df, 'spec', n=3)
                temp_df = cluster_indicators.reset_index()
                temp_df.columns = ['geneID', 'count']
                temp_df['cluster'] = clust_id
                temp_df['type'] = 'neg'
                neg_indicators = pd.concat([neg_indicators, temp_df], ignore_index=True)
                pbar.update(1)
        return neg_indicators
    
    def create_full_indicator_df(self):
        temp_pos = self.positive_indicators.reset_index()
        temp_pos.columns = ['geneID', 'count']
        temp_pos['cluster'] = self.positive_id
        temp_pos['type'] = 'pos'
        indicators_fulldf = pd.concat([temp_pos, self.negative_indicators], ignore_index=True)
        return indicators_fulldf.sort_values(by='count', ascending=False)

class Comparator:
    def __init__(self, 
                 indicator: Indicator,
                 metadata: pd.DataFrame,
                 markers: pd.Series,
                 contours: pd.Series,
                 rep_perc: float = 1.0
                 ):
        self.parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        self.indicator = indicator
        self.origin_csv = self.indicator.origin_csv
        self.annotaion = self.indicator.annotation
        self.metadata = metadata
        self.markers = markers
        self.contours = contours
        self.pos_indicators = self.indicator.positive_indicators 
        self.neg_indicators = self.indicator.negative_indicators
        self.rep_perc = rep_perc
        self.testidx = self.set_testidx() 
        self.thin()

    def set_testidx(self):
        test_length = int(np.ceil(len(self.metadata) * self.rep_perc))
        ids = np.sort(np.random.choice(len(self.metadata), test_length, replace=False))
        return ids
    
    def thin(self):
        self.metadata = self.metadata.iloc[self.testidx]
        self.contours = self.contours.iloc[self.testidx]
        self.markers = self.markers.iloc[self.testidx]


## USE MARKERS FOR COMPARISON 
# ## best way to do it is make a mask for the indicators to subtract from markers array
# avoid np.where

    def create_indicator_mask(self,
                              #origin_img_fname: str,
                              xrange: tuple = None,
                              yrange: tuple = None,
                              indicators: dict = None #key is a string (geneID), value is 1 or -1 to show positive or negative inficator
                              ):
        # needs testing, maybe more optimize 
        mask = np.zeros((xrange[1]-xrange[0], yrange[1]-yrange[0]), dtype=np.int8)
        for ind, value in indicators.items():
            sub_origin = self.origin_csv.iloc[self.origin_csv['geneID']==ind]
            grabs_inds = (
                (xrange[0] <= sub_origin['x']) & (sub_origin['x'] <= xrange[1]) &
                (yrange[0] <= sub_origin['y']) & (sub_origin['y'] <= yrange[1])
            )
            grabs = sub_origin.loc[grabs_inds]
            x_coords = (grabs['x'] - xrange[0]).astype(int).to_numpy()
            y_coords = (grabs['y'] - yrange[0]).astype(int).to_numpy()
            mask[y_coords, x_coords] = value
        return mask

    def create_pos_mask(self):
        # this is a mask for the INDICATORS not for cells
        # might be combined in above function
        pass

    def create_neg_mask(self, 
                        n_indicators: int = 10
                        ):
        # this is a mask for the INDICATORS not for cells
        # might be combined in above function
        pass

    def compare_engine(self):
        pass

    def export_compare_data(self, 
                            out_df: pd.DataFrame = None,
                            ):
        pass



if __name__ == '__main__':
    """contours = test_import_markers(
        '/Users/brianbrogan/Desktop/KI24/SpatialSegmentation/mask_maker_output'
    )
    print(contours.head(10))
    print(contours.shape)
    print(contours.columns)
    print(type(contours.iloc[0]))"""

    #visualize_contours(contours.loc[40, 'contour_arrs'])

    indicatorTester = Indicator(
        origin_csv_fname='/Users/brianbrogan/Desktop/KI24/ClusterImgGen2024/STERSEQ/input/1996-081_GFM_SS200000954BR_A2_tissue_cleaned_cortex_crop.csv',
        annotation_fname='/Users/brianbrogan/Desktop/KI24/ClusterImgGen2024/STERSEQ/output/01. Co-expression network/1996-081_GFM_SS200000954BR_A2_bin100_tissue_cleaned/Cluster_annotation.csv'
        )
    print(indicatorTester.indicators.head(25))

