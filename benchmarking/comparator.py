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
            self.negative_indicators = self.find_neg_indicators_df()
        self.indicators = self.create_full_indicator_df()

    def get_pos_spec_df(self, pos_spec):
        # returns positive indicator dataframe givin list of geneID strings, sets all types to "pos"
        pos_spec_df = pd.DataFrame({"geneID": pos_spec, "type": "pos"})
        return pos_spec_df

    def get_neg_spec_df(self, neg_spec):
        # returns positive indicator dataframe givin list of geneID strings, sets all types to "pos"
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
                          division: bool = False, # If true stratifies neg indicators by cell type
                          ):
        assert indicator_type == 'min_count' or indicator_type == 'spec',\
            "indicator_type must either be 'min_count' or 'spec'."
        indicators = cluster_df['geneID'].value_counts()
        if indicator_type == 'min_count':
            indicators = indicators.loc[indicators>= self.indicator_minimum]
        elif indicator_type == 'spec':
            if division:
                indicators = indicators.iloc[:self.n_neg]
            #elif division is None:
                #indicators = indicators
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
    
    def find_neg_indicators_df(self):
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
            neg_indicators = neg_indicators.sort_values(by='count', ascending=False)
        return neg_indicators.iloc[:self.n_neg]
    
    def create_full_indicator_df(self):
        indicators_full_df = pd.concat([self.positive_indicators, self.negative_indicators], ignore_index=True)
        return indicators_full_df.sort_values(by='count', ascending=False)
    
    def create_indicator_dict(self,
                              indicators: pd.DataFrame):
        temp = indicators[['geneID', 'type']]
        indicator_dict = {row["geneID"]: 1 if row["type"] == "pos" else -1 for _, row in temp.iterrows()}
        return indicator_dict
    
class RandomGrabIndicator(Indicator):
    def __init__(self,
                 origin_csv: pd.DataFrame,
                 annotation: pd.DataFrame,
                 x_win: tuple, # first dimension (cols)
                 y_win: tuple, # second dimension (rows)
                 ):
        super().__init__(origin_csv, annotation)
        self.x_win = x_win
        self.y_win = y_win
        window = (
                (y_win[0] <= origin_csv['x']) & (origin_csv['x'] < y_win[1]) &
                (x_win[0] <= origin_csv['y']) & (origin_csv['y'] < x_win[1])
            )
        print('number of datapoints in window: ', len(window))

        self.most_common_IDs = {
            'NEURON': ['SNAP25', 'PRNP', 'UCHL1', 'TUBB2A', 'NRGN', 'CALM1', 'IDS', 'NEFL', 'VSNL1', 'RTN1', 'THY1', 'ENC1'],
            'ASCTROCYTE': ['CLU', 'SLC1A2', 'MT3', 'AQP4', 'SPARCL1', 'ATP1A2', 'GJA1', 'CPE', 'CST3', 'GLUL', 'SLC1A3', 'MT2A'],
            'OLIGODENDROCYTE': ['PLP1', 'CRYAB', 'SCD', 'CNP', 'QDPR', 'TF', 'MOBP', 'CLDND1', 'SEPTIN4', 'SELENOP', 'MAG', 'CLDN11'],
            'MICROGLIA': ['NLRP1', 'RPS19', 'CTSB', 'CD74', 'FCGBP', 'C3', 'LAPTM5', 'TSPO', 'HLA-DRA', 'C1QA', 'CSF1R', 'BOLA2B'],
            'VASCULATURE': ['CLDN5', 'SLC7A5', 'EGFL7', 'IFITM3', 'VWF', 'FLT1', 'SLC2A1', 'ETS2', 'ITM2A', 'SLC2A3', 'PODXL', 'SLC16A1']
        }
        
        self.non_neuron_IDs = [
            gene for key, values in self.most_common_IDs.items() 
            if key != 'NUERON' 
            for gene in values
        ]

        self.origin_window = origin_csv.loc[window]
        self.pos_win_df = self.pos_window_df()
        self.neg_win_df = self.neg_window_df()
    
    def pos_window_df(self):
        pos_origin_window = self.origin_window[self.origin_window['geneID'].isin(self.most_common_IDs['NEURON'])]
        return pos_origin_window

    def neg_window_df(self):
        neg_origin_window = self.origin_window[self.origin_window['geneID'].isin(self.non_neuron_IDs)]
        return neg_origin_window
    


class Comparator:
    def __init__(self, 
                 metadata: pd.DataFrame,
                 markers: pd.Series,
                 contours: pd.Series,
                 indicator, #Indicator = None,
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
        # add kd trees for x and y columns of origin for faster mask creation
        
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

    def create_indicator_mask_ex(self,
                                 xrange: tuple = None, # first dimension of image file (cols)
                                 yrange: tuple = None, # second dimension of image file (rows)
                                 indicators: dict = None #key is a string (geneID), value is 1 or -1 to show positive or negative inficator
                                 ):
        # can pass positive, negative, or both dictionary as long as format is correct
        # use KDTree eventually
        #self.indicator_mask = None
        mask = np.zeros((yrange[1]-yrange[0], xrange[1]-xrange[0]), dtype=np.int8)
        for indID, value in indicators.items():
            sub_origin = self.origin_csv.loc[self.origin_csv['geneID']==indID]
            grabs_inds = (
                (yrange[0] <= sub_origin['x']) & (sub_origin['x'] < yrange[1]) &
                (xrange[0] <= sub_origin['y']) & (sub_origin['y'] < xrange[1])
                #(xrange[0] <= sub_origin['x']) & (sub_origin['x'] < xrange[1])
            )
            grabs = sub_origin.loc[grabs_inds]
            y_coords = (grabs['x'] - yrange[0]).astype(int).to_numpy()
            x_coords = (grabs['y'] - xrange[0]).astype(int).to_numpy()

            #x_coords = grabs['x'].astype(int).to_numpy()
            #y_coords = grabs['y'].astype(int).to_numpy()

            #x_coords = ((xrange[1] - grabs['x']).astype(int).to_numpy())
            #y_coords = ((yrange[1] - grabs['y']).astype(int).to_numpy())
            mask[y_coords, x_coords] = value
        self.indicator_mask = mask
        return self.indicator_mask
    
    def create_indicator_mask_rand(self,
                                   xrange: tuple, # first dimension of image file (cols)
                                   yrange: tuple, # second dimension of image file (rows)
                                   pos_window_df: pd.DataFrame,
                                   neg_window_df: pd.DataFrame,
                                   density: float = 1.0 # between 0 and 1
                                   ):
        pos_window_df = pos_window_df.reset_index()
        neg_window_df = neg_window_df.reset_index()
        mask = np.zeros((yrange[1]-yrange[0], xrange[1]-xrange[0]), dtype=np.int8)
        n = int(density*min(len(pos_window_df), len(neg_window_df)))
        pos_inds = np.random.randint(0, len(pos_window_df), n)
        neg_inds = np.random.randint(0, len(neg_window_df), n)
        pos_grabs = pos_window_df.loc[pos_inds]
        neg_grabs = neg_window_df.loc[neg_inds]
        pos_y_coords = (pos_grabs['x'] - yrange[0]).astype(int).to_numpy()
        pos_x_coords = (pos_grabs['y'] - xrange[0]).astype(int).to_numpy()
        neg_y_coords = (neg_grabs['x'] - yrange[0]).astype(int).to_numpy()
        neg_x_coords = (neg_grabs['y'] - xrange[0]).astype(int).to_numpy()

        mask[pos_y_coords, pos_x_coords] = 1
        mask[neg_y_coords, neg_x_coords] = -1

        self.indicator_mask = mask
        #print(len(pos_y_coords))
        #print(len(pos_x_coords))
        return self.indicator_mask

            
        
    

    # shit box
    ####################
    def create_indicator_mask_old(self,
                                xrange: tuple = None,  # First dimension of image file (rows)
                                yrange: tuple = None,  # Second dimension of image file (columns)
                                indicators: dict = None  # key: geneID, value: 1 or -1
                                ):
        # Initialize the mask with shape (rows, columns)
        mask = np.zeros((xrange[1] - xrange[0], yrange[1] - yrange[0]), dtype=np.int8)

        for indID, value in indicators.items():
            # Filter data for the given geneID
            sub_origin = self.origin_csv.loc[self.origin_csv['geneID'] == indID]

            # Find data points within the specified ranges
            grabs_inds = (
                (xrange[0] <= sub_origin['y']) & (sub_origin['y'] < xrange[1]) &  # Check row range
                (yrange[0] <= sub_origin['x']) & (sub_origin['x'] < yrange[1])   # Check column range
            )
            grabs = sub_origin.loc[grabs_inds]

            # Calculate coordinates relative to the given window
            x_coords = (grabs['y'] - xrange[0]).astype(int).to_numpy()  # Use 'y' for rows
            y_coords = (grabs['x'] - yrange[0]).astype(int).to_numpy()  # Use 'x' for columns

            # Assign indicator value to the mask
            mask[x_coords, y_coords] = value

        # No transpose is needed since the ranges now directly align with the dimensions
        self.indicator_mask = mask.T
        return self.indicator_mask
    
    def window_indicator_mask(self,
                              full_shape: tuple,
                              xrange: tuple,
                              yrange: tuple,
                              indicators: dict
                              ):
        mask = np.zeros(full_shape, dtype = np.int8)

        for indID, value in indicators.items():
            sub_origin = self.origin_csv.loc[self.origin_csv['geneID'] == indID]
            grabs_inds = (
                (xrange[0] <= sub_origin['y']) & (sub_origin['y'] < xrange[1]) &  # Check row range
                (yrange[0] <= sub_origin['x']) & (sub_origin['x'] < yrange[1])   # Check column range
            )
            grabs = sub_origin.loc[grabs_inds]
            y_coords = (grabs['y']).astype(int).to_numpy()  # Use 'y' for rows
            x_coords = (grabs['x']).astype(int).to_numpy()  # Use 'x' for columns
            mask[y_coords, x_coords] = value

        self.indicator = mask.T
        return self.indicator_mask
    ####################

    def compare_engine(self,
                       #xrange, # first dimension (cols)
                       #yrange, # second dimension (rows)
                       indicator_mask
                       ):
                       #method: str = 'rand'):
        res = pd.DataFrame(columns=['test_paramID', 'mark', 'total', 'n_true_pos', 'n_false_neg'])
        #if method == 'ex':
            #self.create_indicator_mask_ex(xrange, yrange, self.indicator.create_indicator_dict(indicators))
        #elif method == 'rand':
        #self.indicator_mask_rand(xrange, yrange,)
        for i, marker_arr in enumerate(self.markers):
            test_paramID = self.metadata.loc[i, 'test_paramID']
            res = pd.concat([res, self.run_comp_series(indicator_mask, marker_arr, test_paramID)], ignore_index=True)
        return res

    def run_comp_series(self,
                        ind_mask: np.ndarray,
                        marker_array: np.ndarray,
                        test_paramID: str
                        ):
        res = pd.DataFrame(columns=['test_paramID', 'mark', 'total', 'n_true_pos', 'n_false_neg'])
        targets = np.unique(marker_array)[1:]
        ## FIX THIS LATER - INVOLVES REP PERC
        #test_len = int(np.ceil(len(targets) * self.rep_perc))
        #test_len = min(test_len, len(targets))
        #test_idx = np.sort(np.random.choice(len(self.metadata), test_len, replace=False))
        #targets = targets[test_idx]
        for target in targets:
            total, tp, fn = self.count_marker_and_indicators(ind_mask,
                                                             marker_array,
                                                             target)
            toAdd = {'test_paramID': test_paramID,
                     'mark': target,
                     'total': total,
                     'n_true_pos': tp,
                     'n_false_neg': fn}
            #res = pd.concat([res, toAdd])
            res.loc[len(res)] = toAdd
        return res
            

    def count_marker_and_indicators(self,
                                    indicator_mask,
                                    marker_array,
                                    target
                                    ):
        #print("indicator mask shape: ", ind_mask.shape)
        #print("marker array shape: ", marker_array.shape)
        #assert ind_mask.shape == marker_array.shape,\
        #"problemo"
        mask = marker_array == target
        total_count = np.sum(mask)
        #print('total count: ', total_count)
        positive_count = np.sum(indicator_mask[mask] == 1)
        negative_count = np.sum(indicator_mask[mask] == -1)
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

    indic_mask = comparatorTester.create_indicator_mask(xrange = x_win, 
                                                            yrange = y_win,
                                                            indicators = indicatorTester.create_indicator_dict(indicatorTester.indicators))
    
    print('done.', flush=True)
    
    print(np.unique(indic_mask))
    print(indic_mask.shape)

    #plt.imshow(indic_mask)

