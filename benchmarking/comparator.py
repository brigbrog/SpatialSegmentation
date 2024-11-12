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
        self.origin_df = self.import_origin()
        self.pos_indicators = pos_indicators # take from indicatorFinder DONE
        self.neg_indicators = neg_indicators # take from indicatorFinder DONE
        self.rep_perc = rep_perc
        # use tqdm to display importation tasks #
        self.testidx = self.set_testidx() 
        self.metadata = self.import_metadata() # read parquet -> DataFrame 
        self.contours = self.import_contours() # read json -> DataFrame
        self.markers = self.import_markers() # read json -> DataFrame
    
    def import_origin(self):
        origin_df = pd.read_csv(self.origin_csv_fname)
        return origin_df
        
    def import_metadata(self):
        metadata = pd.read_parquet(self.pqfname, engine='fastparquet')
        return metadata

    def import_contours(self):
        pass 

    def import_markers(self):
        pass

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


class IndicatorFinder:
    def __init__(self,
                 origin_csv_fname: str = None,
                 annotation_fname: str = None,
                 indicator_minimum: int = 30000,
                 positive_id: int = 0,
                 negative_ids: int|list = [4,5,11,13]
                 ):
        self.origin_csv = pd.read_csv(origin_csv_fname)
        self.annotation = pd.read_csv(annotation_fname)
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


def test_import_metadata(pqfname):
    metadata = pd.read_parquet(pqfname, engine='fastparquet')
    return metadata



if __name__ == '__main__':
    #metadata = test_import_metadata(
    #    '/Users/brianbrogan/Desktop/KI24/SpatialSegmentation/mask_maker_output/variable_segmentation_metadata.parquet'
    #)
    #print(metadata.head(10))

    indicatorTester = IndicatorFinder(
        origin_csv_fname='/Users/brianbrogan/Desktop/KI24/ClusterImgGen2024/STERSEQ/input/1996-081_GFM_SS200000954BR_A2_tissue_cleaned_cortex_crop.csv',
        annotation_fname='/Users/brianbrogan/Desktop/KI24/ClusterImgGen2024/STERSEQ/output/01. Co-expression network/1996-081_GFM_SS200000954BR_A2_bin100_tissue_cleaned/Cluster_annotation.csv'
        )
    print(indicatorTester.indicators.head(25))

