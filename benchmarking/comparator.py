import numpy as np
import pandas as pd
import os
from tqdm import tqdm
from manager import DataManager

class RandomGrabIndicator:
    def __init__(self,
                 origin_csv: pd.DataFrame,
                 x_win: tuple, # first dimension (cols)
                 y_win: tuple, # second dimension (rows)
                 equalize: bool = True
                 ):
        '''Constructor for RandomGrabIndicator. Organizes origin data and window for analysis.
        Params:
            origin_csv: (pandas.DataFrame) The dataframe to which the origin data is stored.
            x_win: (tuple) The first dimension of the image slice for analysis, the columns of the image array.
            y_win: (tuple) The second dimension of the image slice for analysis, the rows of the image array.
            equalize: (bool) boolean controlling the equalization of postive and negative indicator dataframes. 
                      If True, both are equalized to the minimum length. Default is True.'''
        
        self.origin_csv = origin_csv
        self.x_win = x_win
        self.y_win = y_win
        window = (
                (y_win[0] <= origin_csv['x']) & (origin_csv['x'] < y_win[1]) &
                (x_win[0] <= origin_csv['y']) & (origin_csv['y'] < x_win[1])
            )
        self.most_common_IDs = {
            'NEURON': ['MT-RNR2','SNAP25', 'PRNP', 'UCHL1', 'TUBB2A', 'NRGN', 'CALM1', 'IDS', 'NEFL', 'VSNL1', 'RTN1', 'THY1', 'ENC1', 'MT-CO2', 'MT-TV', 'MT-CO3', 'MT-ND1'],
            'ASCTROCYTE': ['CLU', 'SLC1A2', 'MT3', 'AQP4', 'SPARCL1', 'ATP1A2', 'GJA1', 'CPE', 'CST3', 'GLUL', 'SLC1A3', 'MT2A'],
            'OLIGODENDROCYTE': ['PLP1', 'CRYAB', 'SCD', 'CNP', 'QDPR', 'TF', 'MOBP', 'CLDND1', 'SEPTIN4', 'SELENOP', 'MAG', 'CLDN11'],
            'MICROGLIA': ['NLRP1', 'RPS19', 'CTSB', 'CD74', 'FCGBP', 'C3', 'LAPTM5', 'TSPO', 'HLA-DRA', 'C1QA', 'CSF1R', 'BOLA2B'],
            'VASCULATURE': ['CLDN5', 'SLC7A5', 'EGFL7', 'IFITM3', 'VWF', 'FLT1', 'SLC2A1', 'ETS2', 'ITM2A', 'SLC2A3', 'PODXL', 'SLC16A1']
        }
        self.non_neuron_IDs = self.most_common_IDs['ASCTROCYTE'] + self.most_common_IDs['OLIGODENDROCYTE'] + self.most_common_IDs['MICROGLIA'] + self.most_common_IDs['VASCULATURE']
        self.origin_window = origin_csv.loc[window]
        self.pos_win_df = self.pos_window_df()
        self.neg_win_df = self.neg_window_df()
        chop = np.min((len(self.pos_win_df), len(self.neg_win_df)))
        if equalize:
            self.pos_win_df = self.pos_win_df.sample(frac=1).reset_index(drop=True).loc[:chop]
            self.neg_win_df = self.neg_win_df.sample(frac=1).reset_index(drop=True).loc[:chop]
    
    def pos_window_df(self):
        '''Generator method for postive window gene indicator DataFrame. Pulls all instances in origin data from Neuron 
        indicator list within object analysis window.
        Params:
            None
        Returns: 
            pos_origin_window: (pandas.DataFrame) The postive indicators with x, y coordinates within object window.
        '''
    
        pos_origin_window = self.origin_window[self.origin_window['geneID'].isin(self.most_common_IDs['NEURON'])]
        return pos_origin_window

    def neg_window_df(self):
        '''Generator method for negative window gene indicator DataFrame. Pulls all instances in origin data from non Neuron 
        indicator list within object analysis window.
        Params:
            None
        Returns: 
            neg_origin_window: (pandas.DataFrame) The negative indicators with x, y coordinates within object window.
        '''
    
        neg_origin_window = self.origin_window[self.origin_window['geneID'].isin(self.non_neuron_IDs)]
        return neg_origin_window

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
        '''Generates a DataFrame of positive indicators from a given list of gene IDs.
        Params:
            pos_spec: (list) List of gene IDs to classify as positive indicators.
        Returns:
            pos_spec_df: (pandas.DataFrame) DataFrame with gene IDs and their classification as positive.
        '''
        pos_spec_df = pd.DataFrame({"geneID": pos_spec, "type": "pos"})
        return pos_spec_df

    def get_neg_spec_df(self, neg_spec):
        '''Generates a DataFrame of negative indicators from a given list of gene IDs.
        Params:
            neg_spec: (list) List of gene IDs to classify as negative indicators.
        Returns:
            neg_spec_df: (pandas.DataFrame) DataFrame with gene IDs and their classification as negative.
        '''
        neg_spec_df = pd.DataFrame({"geneID": neg_spec, "type": "pos"})
        return neg_spec_df

    def get_gene_set(self,
                      id: int = None):
        '''Filtres annotated dataframe for specified cluster id and returns genes as a set.
        Params:
            id: (int) Cluster index for desired gene set. 
        Returns:
            clust_genes: (set) Set of gene names from annotation dataframe reference for specified cluster id.'''
        
        assert id is not None, \
            "Cluster ID must be provided to pull indicators."
        clust = self.annotation.loc[self.annotation['cluster']==id, :]
        clust_genes = set(np.unique(clust['features']))
        return clust_genes
    
    def create_cluster_df(self,
                          gene_set: pd.Series = None
                          ):
        '''Returns pandas.DataFrame for specified gene name set. Pulled from origin data.
        Params:
            gene_set: set of gene generated by get_gene_set for specified cluster id.
        Returns: 
            cluster_df: instances pulled from origin data for specified set of genes.'''
        
        cluster_df = self.origin_csv[self.origin_csv['geneID'].isin(gene_set)]
        return cluster_df
        
    def filter_indicators(self, 
                          cluster_df: pd.DataFrame = None,
                          indicator_type: str = 'min_count'
                          ):
        '''Filters cluster dataframe (returned from create_cluster_df) by count minimum or for first n instances.
        Params:
            cluster_df: (pd.DataFrame) cluster dataframe to be filtered.
            indicator_type: (str) filtering method applied to indicators. "min_count" for minimum count and "spec" for top n instances.
        Returns:
            indicators: (pd.DataFrame) filtered dataframe of indicator instances drawn from cluster_df.'''
        
        assert indicator_type == 'min_count' or indicator_type == 'spec',\
            "indicator_type must either be 'min_count' or 'spec'."
        indicators = cluster_df['geneID'].value_counts()
        if indicator_type == 'min_count':
            indicators = indicators.loc[indicators>= self.indicator_minimum]
        elif indicator_type == 'spec':
            indicators = indicators.iloc[:self.n_neg]
        return indicators
    
    def find_pos_indicators_df(self):
        '''Verbosely retrieves genes associated with the positive cluster, processes to cluster-level DataFrame, applies filtering criteria to find positive indicator genes.
        Params:
            None
        Returns:
            pos_inds: (pd.DataFrame) A DataFrame with columns ['geneID', 'count', 'cluster', 'type']
            containing the positive indicator genes.'''
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
        '''Verbosely retrieves genes associated with the positive cluster, processes to cluster-level Series, applies filtering criteria to find positive indicator genes.
        Params:
            None
        Returns:
            pos_indicators: (pd.Series) Containing the positive indicator genes.'''
        with tqdm(total=3, desc="Finding positive indicators", unit="step") as pbar:
            pos_genes = self.get_gene_set(self.positive_id)
            pbar.update(1)
            pos_cluster_df = self.create_cluster_df(pos_genes)
            pbar.update(1)
            pos_indicators = self.filter_indicators(pos_cluster_df)
            pbar.update(1)
        return pos_indicators
    
    def find_neg_indicators_df(self):
        '''Verbosely retrieves genes associated with the negative cluster, processes to cluster-level DataFrame, applies filtering criteria to find negative indicator genes.
        Params:
            None
        Returns:
            (pd.DataFrame) A DataFrame with columns ['geneID', 'count', 'cluster', 'type']
            containing the negative indicator genes.'''
        neg_indicators = pd.DataFrame(columns=['geneID', 'count', 'cluster', 'type'])
        with tqdm(total=len(self.negative_ids), desc="Finding negative indicators", unit="cluster") as pbar:
            for clust_id in self.negative_ids:
                clust_genes = self.get_gene_set(clust_id)
                cluster_df = self.create_cluster_df(clust_genes)
                cluster_indicators = self.filter_indicators(cluster_df, 'spec')#, division=True)
                temp_df = cluster_indicators.reset_index()
                temp_df.columns = ['geneID', 'count']
                temp_df['cluster'] = clust_id
                temp_df['type'] = 'neg'
                neg_indicators = pd.concat([neg_indicators, temp_df], ignore_index=True)
                pbar.update(1)
            neg_indicators = neg_indicators.sort_values(by='count', ascending=False)
        return neg_indicators.iloc[:self.n_neg]
    
    def create_full_indicator_df(self):
        '''Concatenates saved postive and negative cluster dataframes. Returns concatenation.'''
        indicators_full_df = pd.concat([self.positive_indicators, self.negative_indicators], ignore_index=True)
        return indicators_full_df.sort_values(by='count', ascending=False)
    
    def create_indicator_dict(self,
                              indicators: pd.DataFrame):
        '''Creates a dictionary mapping gene IDs to their indicator type (positive or negative).
        Params:
            indicators: (pandas.DataFrame) DataFrame containing gene IDs and their types ('pos' or 'neg').
        Returns:
            indicator_dict: (dict) Dictionary mapping each gene ID to 1 (positive) or -1 (negative).
        '''
        temp = indicators[['geneID', 'type']]
        indicator_dict = {row["geneID"]: 1 if row["type"] == "pos" else -1 for _, row in temp.iterrows()}
        return indicator_dict
    
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

    def create_indicator_mask_ex(self,
                                 xrange: tuple = None, # first dimension of image file (cols)
                                 yrange: tuple = None, # second dimension of image file (rows)
                                 indicators: dict = None #key is a string (geneID), value is 1 or -1 to show positive or negative indicator
                                 ):
        '''Generates a mask array based on indicator gene IDs for a specified region of the image.
        Params:
            xrange: (tuple) Range of columns (x-dimension) in the image.
            yrange: (tuple) Range of rows (y-dimension) in the image.
            indicators: (dict) Dictionary mapping gene IDs to 1 (positive) or -1 (negative).
        Returns:
            mask: (numpy.ndarray) Mask array with values 1 (positive), -1 (negative), or 0 (no indicator).
        '''
        ## Suggest implementing KDTree or other more efficient point searching algorithm
        mask = np.zeros((yrange[1]-yrange[0], xrange[1]-xrange[0]), dtype=np.int8)
        for indID, value in indicators.items():
            sub_origin = self.origin_csv.loc[self.origin_csv['geneID']==indID]
            grabs_inds = (
                (yrange[0] <= sub_origin['x']) & (sub_origin['x'] < yrange[1]) &
                (xrange[0] <= sub_origin['y']) & (sub_origin['y'] < xrange[1])
            )
            grabs = sub_origin.loc[grabs_inds]
            y_coords = (grabs['x'] - yrange[0]).astype(int).to_numpy()
            x_coords = (grabs['y'] - xrange[0]).astype(int).to_numpy()
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
        '''Generates a randomized mask array based on positive and negative indicators within a region.
        Params:
            xrange: (tuple) Range of columns (x-dimension) in the image.
            yrange: (tuple) Range of rows (y-dimension) in the image.
            pos_window_df: (pandas.DataFrame) DataFrame containing positive indicator coordinates.
            neg_window_df: (pandas.DataFrame) DataFrame containing negative indicator coordinates.
            density: (float) Fraction of available indicators to include in the mask, between 0 and 1.
        Returns:
            mask: (numpy.ndarray) Mask array with randomized positive and negative indicators.
        '''
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
        return self.indicator_mask

    def compare_engine(self,
                       indicator_mask: np.ndarray
                       ):
        '''Compares an indicator mask to marker arrays and generates a summary DataFrame.
        Params:
            indicator_mask: (numpy.ndarray) Mask array with positive and negative indicators.
        Returns:
            comparison_df: (pandas.DataFrame) DataFrame summarizing comparison results for each marker.
        '''
        res = pd.DataFrame(columns=['test_paramID', 'mark', 'total', 'n_true_pos', 'n_false_neg'])
        for i, marker_arr in enumerate(self.markers):
            test_paramID = self.metadata.loc[i, 'test_paramID']
            res = pd.concat([res, self.run_comp_series(indicator_mask, marker_arr, test_paramID)], ignore_index=True)
        self.comparison_df = res
        return self.comparison_df

    def run_comp_series(self,
                        ind_mask: np.ndarray,
                        marker_array: np.ndarray,
                        test_paramID: str
                        ):
        '''Processes a single marker array to compare against the indicator mask.
        Params:
            ind_mask: (numpy.ndarray) Mask array with positive and negative indicators.
            marker_array: (numpy.ndarray) Array containing marker regions.
            test_paramID: (str) Identifier for the test parameters being analyzed.
        Returns:
            res: (pandas.DataFrame) DataFrame summarizing counts of true positives and false negatives.
        '''
        res = pd.DataFrame(columns=['test_paramID', 'mark', 'total', 'n_true_pos', 'n_false_neg'])
        targets = np.unique(marker_array)[1:]
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
                                    indicator_mask: np.ndarray,
                                    marker_array: np.ndarray,
                                    target: int
                                    ):
        '''Counts total markers, true positives, and false negatives for a given target.
        Params:
            indicator_mask: (numpy.ndarray) Mask array with positive and negative indicators.
            marker_array: (numpy.ndarray) Array containing marker regions.
            target: (int) Marker value to analyze.
        Returns:
            total_count: (int) Total number of markers of the given target.
            positive_count: (int) Number of true positive markers.
            negative_count: (int) Number of false negative markers.
        '''
        mask = marker_array == target
        total_count = np.sum(mask)
        positive_count = np.sum(indicator_mask[mask] == 1)
        negative_count = np.sum(indicator_mask[mask] == -1)
        return total_count, positive_count, negative_count

    def export_compare_data(self, 
                            out_dir: str = None,
                            ):
        '''Exports the comparison DataFrame to a CSV file.
        Params:
            out_dir: (str) Directory path to save the CSV file.
        Returns:
            None
        '''
        assert self.comparison_df is not None, "Run compare engine before exporting compare data."
        self.comparison_df.to_csv(out_dir, index=False)

    def rgb_indicator_mask(self,
                           mask: np.ndarray):
        '''Generates an RGB image representation array of the indicator mask.
        Params:
            mask: (numpy.ndarray) Mask array with positive and negative indicators.
        Returns:
            rgb: (numpy.ndarray) RGB image array where positive indicators are green, negative are red, and others are black.
        '''
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

