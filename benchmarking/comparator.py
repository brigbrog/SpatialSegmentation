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

from manager import DataManager

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
    def __init__(self,
                 origin_csv: pd.DataFrame,
                 annotation: pd.DataFrame,
                 specify: bool = False,
                 pos_spec: list = None,
                 neg_spec: list = None,
                 indicator_minimum: int = 30000,
                 n_neg: int = 10,
                 positive_id: int = 0,
                 negative_ids: int|list = [4,5,11,13]
                 ):
        self.origin_csv = origin_csv
        self.annotation = annotation
        self.indicator_minimum = indicator_minimum
        self.n_neg = n_neg
        # currently number of each cluster, maybe change to total ?
        self.positive_id = positive_id
        self.negative_ids = negative_ids
        if specify:
            assert pos_spec is not None and neg_spec is not None, \
            "pos_spec and neg_spec must be populated for specified indicator DataFrame creation."
            self.postive_indicators = self.get_pos_spec_df(pos_spec)
            self.negative_indicators = self.get_neg_spec_df(neg_spec)
        elif not specify: 
            self.positive_indicators = self.find_pos_indicators_df()
            self.negative_indicators = self.find_neg_indicators_df(self.n_neg)
        self.indicators = self.create_full_indicator_df()

    def get_pos_spec_df(self, pos_spec):
        pos_spec_df = pd.DataFrame({"geneID": pos_spec, "type": "pos"})
        return pos_spec_df

    def get_neg_spec_df(self, neg_spec):
        neg_spec_df = pd.DataFrame({"geneID": neg_spec, "type": "pos"})
        return neg_spec_df

    def get_gene_set(self,
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
                          indicator_type: str = 'min_count', #'min_count' or 'spec'
                          ):
        assert indicator_type == 'min_count' or indicator_type == 'spec',\
            "indicator_type must either be 'min_count' or 'spec'."
        indicators = cluster_df['geneID'].value_counts()
        if indicator_type == 'min_count':
            indicators = indicators.loc[indicators>= self.indicator_minimum]
        elif indicator_type == 'spec':
            indicators = indicators.iloc[:self.n_neg]
        return indicators
    
    def find_pos_indicators_df(self):
        with tqdm(total=3, desc="Finding positive indicators", unit="step") as pbar:
            pos_genes = self.get_gene_set(self.positive_id)
            pbar.update(1)
            pos_df = self.create_cluster_df(pos_genes)
            pbar.update(1)
            pos_inds = self.filter_indicators(pos_df, 'min_count').reset_index()
            pbar.update(1)
        pos_inds.columns = ['geneID', 'count']
        pos_inds['cluster'] = self.positive_id
        pos_inds['type'] = 'pos'
        return pos_inds
    
    def find_pos_indicators_series(self):
        with tqdm(total=3, desc="Finding positive indicators", unit="step") as pbar:
            pos_genes = self.get_gene_set(self.positive_id)
            pbar.update(1)
            pos_cluster_df = self.create_cluster_df(pos_genes)
            pbar.update(1)
            pos_indicators = self.filter_indicators(pos_cluster_df)
            pbar.update(1)
        return pos_indicators
    
    def find_neg_indicators_df(self, n):
        neg_indicators = pd.DataFrame(columns=['geneID', 'count', 'cluster', 'type'])
        with tqdm(total=len(self.negative_ids), desc="Finding negative indicators", unit="cluster") as pbar:
            for clust_id in self.negative_ids:
                clust_genes = self.get_gene_set(clust_id)
                cluster_df = self.create_cluster_df(clust_genes)
                cluster_indicators = self.filter_indicators(cluster_df, 'spec')
                temp_df = cluster_indicators.reset_index()
                temp_df.columns = ['geneID', 'count']
                temp_df['cluster'] = clust_id
                temp_df['type'] = 'neg'
                neg_indicators = pd.concat([neg_indicators, temp_df], ignore_index=True)
                pbar.update(1)
        return neg_indicators
    
    def create_full_indicator_df(self):
        indicators_full_df = pd.concat([self.positive_indicators, self.negative_indicators], ignore_index=True)
        return indicators_full_df.sort_values(by='count', ascending=False)
    
    def create_indicator_dict(self,
                              indicators: pd.DataFrame):
        temp = indicators[['geneID', 'type']]
        indicator_dict = {row["geneID"]: 1 if row["type"] == "pos" else -1 for _, row in temp.iterrows()}
        return indicator_dict

class Comparator:
    def __init__(self, 
                 metadata: pd.DataFrame,
                 markers: pd.Series,
                 contours: pd.Series,
                 indicator: Indicator = None,
                 rep_perc: float = 1.0,
                 ):
        self.parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        self.indicator = indicator
        self.origin_csv = self.indicator.origin_csv
        self.annotaion = self.indicator.annotation
        self.pos_indicators = self.indicator.positive_indicators
        self.neg_indicators = self.indicator.negative_indicators
        self.metadata = metadata
        self.markers = markers
        self.contours = contours
        self.rep_perc = rep_perc
        self.indicator_mask = None
        self.comparison_df = None
        
        #self.testidx = self.set_testidx() 
        #self.thin()

    def set_testidx(self):
        # wrong, should not be slicing metadata, only markers/contours 1 level down
        test_length = int(np.ceil(len(self.metadata) * self.rep_perc))
        ids = np.sort(np.random.choice(len(self.metadata), test_length, replace=False))
        return ids
    
    def thin(self):
        # wrong, should not be slicing metadata, only markers/contours 1 level down
        self.metadata = self.metadata.iloc[self.testidx]
        self.contours = self.contours.iloc[self.testidx]
        self.markers = self.markers.iloc[self.testidx]

    
## USE MARKERS FOR COMPARISON 
# ## best way to do it is make a mask for the indicators to subtract from markers array
# avoid np.where

    def create_indicator_mask(self,
                              xrange: tuple = None,
                              yrange: tuple = None,
                              indicators: dict = None #key is a string (geneID), value is 1 or -1 to show positive or negative inficator
                              ):
        ## MASK INITIALIZATION IS FUCKED UP ##
        # can pass positive, negative, or both dictionary as long as format is correct
        mask = np.zeros(((xrange[1])-xrange[0], (yrange[1])-yrange[0]), dtype=np.int8)
        for ind, value in indicators.items():
            sub_origin = self.origin_csv.loc[self.origin_csv['geneID']==ind]
            grabs_inds = (
                (xrange[0] <= sub_origin['x']) & (sub_origin['x'] <= xrange[1]) &
                (yrange[0] <= sub_origin['y']) & (sub_origin['y'] <= yrange[1])
            )
            grabs = sub_origin.loc[grabs_inds]
            #x_coords = (grabs['x'] - xrange[0]).astype(int).to_numpy()
            #y_coords = (grabs['y'] - yrange[0]).astype(int).to_numpy()
            x_coords = ((xrange[1] - grabs['x']).astype(int).to_numpy())
            y_coords = ((yrange[1] - grabs['y']).astype(int).to_numpy())
            mask[y_coords, x_coords] = value
        self.indicator_mask = mask
        return mask
    

    def compare_engine(self,
                       xrange,
                       yrange, 
                       indicators):
        res = pd.DataFrame(columns=['test_paramID', 'mark', 'total', 'n_true_pos', 'n_false_neg'])
        self.create_indicator_mask(xrange, yrange, indicators)
        for i, marker_arr in enumerate(self.markers):
            test_paramID = self.metadata.loc[i, 'test_paramID']
            pd.concat([res, self.run_comp_series(self.indicator_mask, marker_arr, test_paramID)], ignore_index=True)
        return 

    def run_comp_series(self,
                        ind_mask: np.ndarray,
                        marker_array: np.ndarray,
                        test_paramID: str
                        ):
        res = pd.DataFrame(columns=['test_paramID', 'mark', 'total', 'n_true_pos', 'n_false_neg'])
        targets = np.unique(marker_array)[1:]
        test_len = int(np.ceil(len(targets) * self.rep_perc))
        test_idx = np.sort(np.random.choice(len(self.metadata), test_len, replace=False))
        targets = targets[test_idx]
        for target in targets:
            total, tp, fn = self.count_marker_and_indicators(ind_mask,
                                                             marker_array,
                                                             target)
            toAdd = {'test_paramID': test_paramID,
                     'mark': target,
                     'total': total,
                     'n_true_pos': tp,
                     'n_false_neg': fn}
            res.loc[len(res)] = toAdd
        return res
            

    def count_marker_and_indicators(self,
                                    ind_mask,
                                    marker_array,
                                    target
                                    ):
        print(ind_mask.shape)
        print(marker_array.shape)
        #assert ind_mask.shape == marker_array.shape,\
        #"problemo"
        mask = marker_array == target
        total_count = np.sum(mask)
        print('total count: ', total_count)
        positive_count = np.sum(ind_mask[mask] == 1)
        negative_count = np.sum(ind_mask[mask] == -1)
        return total_count, positive_count, negative_count

        




    def export_compare_data(self, 
                            out_df: pd.DataFrame = None,
                            ):
        pass

    def rgb_indicator_mask(self,
                           mask: np.ndarray):
        rgb = np.zeros((*mask.shape, 3), dtype=np.uint8)
        rgb[mask == -1] = [255, 0, 0]
        rgb[mask == 1] = [0, 255, 0]
        return rgb


if __name__ == '__main__':
    """contours = test_import_markers(
        '/Users/brianbrogan/Desktop/KI24/SpatialSegmentation/mask_maker_output'
    )
    print(contours.head(10))
    print(contours.shape)
    print(contours.columns)
    print(type(contours.iloc[0]))"""

    #visualize_contours(contours.loc[40, 'contour_arrs'])

    print("creating manager...", flush=True)
    manager = DataManager(
        origin_csv_fname='/Users/brianbrogan/Desktop/KI24/ClusterImgGen2024/STERSEQ/input/1996-081_GFM_SS200000954BR_A2_tissue_cleaned_cortex_crop.csv',
        annotation_fname='/Users/brianbrogan/Desktop/KI24/ClusterImgGen2024/STERSEQ/output/01. Co-expression network/1996-081_GFM_SS200000954BR_A2_bin100_tissue_cleaned/Cluster_annotation.csv'
    )
    print("done.")

    indicatorTester = Indicator(
        origin_csv = manager.origin_csv,
        annotation = manager.annotation
        )
    print(indicatorTester.indicators.head(25))

    print("creating indicator mask...", flush=True)
    comparatorTester = Comparator(
        indicator = indicatorTester,
        metadata = manager.metadata,
        markers = manager.markers,
        contours = manager.contours
    )

    y_win = (6000, 7000)
    x_win = (9000, 10000)

    neg_indic_mask = comparatorTester.create_indicator_mask(xrange = x_win, 
                                                            yrange = y_win,
                                                            indicators = indicatorTester.create_indicator_dict(indicatorTester.negative_indicators))
    
    print('done.', flush=True)
    
    print(np.unique(neg_indic_mask))

    plt.imshow(neg_indic_mask)

