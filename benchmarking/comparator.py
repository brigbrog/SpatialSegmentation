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

import gc
from joblib import Parallel, delayed

# git ignore for input output dir DONE
# need to find how to get indicator lists DONE
# FIX MARKERS and CONTOURS importers, they are fucked up
    #contours DONE
    #markers
# create pos/neg masks for quick comparison to origin
    # Only have cell specific true pos false neg, false pos and true neg are rates bc no cell assignment
    # create numpy mask for each contour and pull positive coords or find better way...
    # use indicatorFinder object (maybe attribute ?) DONE
    # remember representative percentage

class Indicator:
    def __init__(self,
                 origin_csv_fname: str = None,
                 annotation_fname: str = None,
                 indicator_minimum: int = 30000,
                 positive_id: int = 0,
                 negative_ids: int|list = [4,5,11,13]
                 ):
        self.origin_csv_fname = origin_csv_fname
        self.annotation_fname = annotation_fname
        self.origin_csv = pd.read_csv(self.origin_csv_fname)
        self.annotation = pd.read_csv(self.annotation_fname)
        self.indicator_minimum = indicator_minimum
        self.positive_id = positive_id
        self.negative_ids = negative_ids
        self.positive_indicators = self.find_pos_indicators()
        self.negative_indicators = self.find_negative_indicators()
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
    
    def find_negative_indicators(self):
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
                 indicator: Indicator = None, 
                 pqfname: str = None,
                 rep_perc: float = 1.0
                 ):
        self.indicator = indicator
        self.origin_csv_fname = self.indicator.origin_csv_fname
        #self.origin_df = self.import_origin()
        self.origin_df = self.indicator.origin_csv
        self.pqfname = pqfname
        self.pos_indicators = self.indicator.pos_indicators 
        self.neg_indicators = self.indicator.neg_indicators
        self.rep_perc = rep_perc
        # use tqdm to display importation tasks #
        self.testidx = self.set_testidx() 
        self.metadata = self.import_metadata() # read parquet -> DataFrame
        self.markers = self.import_markers() # read json -> DataFrame
        self.contours = self.import_contours() # read json -> DataFrame
    
    '''def import_origin(self):
        origin_df = pd.read_csv(self.origin_csv_fname)
        return origin_df'''
        
    def import_metadata(self):
        metadata = pd.read_parquet(self.pqfname, engine='fastparquet')
        return metadata
    
    def import_markers(self, 
                        in_dir: str = 'mask_maker_output'
                        ):
        temp = []
        markers_path = os.path.join(in_dir, "markers.json")
        with open(markers_path, 'r') as f:
            markers_data = json.load(f)
        for iseg_mark_list in markers_data:
            temp.append([np.array(mark, dtype=np.int32) for mark in iseg_mark_list if len(mark)>1])
        mark_df = pd.DataFrame({'marker_arrs': temp}, dtype='object')
        return mark_df

    def import_contours(self, 
                        in_dir: str = 'mask_maker_output'
                        ):
        temp = []
        contours_path = os.path.join(in_dir, "contours.json")
        with open(contours_path, 'r') as f:
            contours_data = json.load(f)
        for iseg_cont_list in contours_data:
            temp.append([np.array(cont, dtype=np.int32) for cont in iseg_cont_list if len(cont)>1])
        cont_df = pd.DataFrame({'contour_arrs': temp}, dtype='object')
        return cont_df

    def set_testidx(self):
        pass

    def create_pos_mask(self):
        pass

    def create_neg_mask(self, 
                        n_indicators: int = 10
                        ):
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


def view_contour(contour_series, idx):
    contours = contour_series.iloc[idx]
    x_vals, y_vals = zip(*[point[0] for point in contours])
    max_x, max_y = max(x_vals), max(y_vals)
    image = np.zeros((max_y + 10, max_x + 10), dtype=np.uint8)
    cv2.drawContours(image, contours, -1, 255, thickness=1)
    plt.imshow(image, cmap='gray')
    plt.title("Contour Visualization")
    plt.axis("off")
    plt.show()

def visualize_contours(contours):
    # Initialize min and max values with large/small extremes
    min_x, min_y = float('inf'), float('inf')
    max_x, max_y = -float('inf'), -float('inf')

    for i, contour in enumerate(contours):
        print(i)
        #print(contour.shape)
        # Extract x and y values for each contour
        x_vals, y_vals = zip(*[point[0] for point in contour])  # Extract x, y pairs
        
        # Update min/max values
        min_x, min_y = min(min_x, min(x_vals)), min(min_y, min(y_vals))
        max_x, max_y = max(max_x, max(x_vals)), max(max_y, max(y_vals))

    # Create an image of the required size
    image = np.zeros((max_y - min_y + 10, max_x - min_x + 10), dtype=np.uint8)
    
    # Draw each contour (shifted to fit in the image)
    for contour in contours:
        #contour_shifted = np.array([[pt[0] - min_x, pt[1] - min_y] for pt in contour], dtype=np.int32)
        contour_shifted = np.array([[pt[0][0] - min_x, pt[0][1] - min_y] for pt in contour], dtype=np.int32)
        cv2.drawContours(image, [contour_shifted], -1, 255, thickness=1)
    
    # Plot the image
    plt.imshow(image, cmap='gray')
    plt.title("Contours Visualization")
    plt.axis('off')
    plt.show()

def test_import_metadata(pqfname):
    metadata = pd.read_parquet(pqfname, engine='fastparquet')
    return metadata

def test_import_contours(in_dir: str = None):
    contours = []
    contours_path = os.path.join(in_dir, "contours.json")
    with open(contours_path, 'r') as f:
        contours_data = json.load(f)
    for iseg_cont_list in contours_data:
        contours.append([np.array(cont, dtype=np.int32) for cont in iseg_cont_list if len(cont)>1])
    cont_df = pd.DataFrame({'contour_arrs': contours}, dtype='object')
    return cont_df

def test_import_markers(in_dir: str = 'mask_maker_output'):
    temp = []
    markers_path = os.path.join(in_dir, "markers.json")
    with open(markers_path, 'r') as f:
        markers_data = json.load(f)
    for iseg_mark_list in markers_data:
        temp.append([np.array(mark, dtype=np.int32) for mark in iseg_mark_list if len(mark)>1])
    mark_df = pd.DataFrame({'marker_arrs': temp}, dtype='object')
    return mark_df


if __name__ == '__main__':
    contours = test_import_markers(
        '/Users/brianbrogan/Desktop/KI24/SpatialSegmentation/mask_maker_output'
    )
    print(contours.head(10))
    print(contours.shape)
    print(contours.columns)
    print(type(contours.iloc[0]))

    #visualize_contours(contours.loc[40, 'contour_arrs'])

    '''indicatorTester = Indicator(
        origin_csv_fname='/Users/brianbrogan/Desktop/KI24/ClusterImgGen2024/STERSEQ/input/1996-081_GFM_SS200000954BR_A2_tissue_cleaned_cortex_crop.csv',
        annotation_fname='/Users/brianbrogan/Desktop/KI24/ClusterImgGen2024/STERSEQ/output/01. Co-expression network/1996-081_GFM_SS200000954BR_A2_bin100_tissue_cleaned/Cluster_annotation.csv'
        )
    print(indicatorTester.indicators.head(25))'''

