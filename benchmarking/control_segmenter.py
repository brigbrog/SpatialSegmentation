import numpy as np
import pandas as pd
import cv2 
from scipy.ndimage import gaussian_filter
import tifffile as tiff
from tqdm import tqdm
import os
import json
import fastparquet

#fix preprocess thing DONE
#add edge contour filter maybe
#omit the job splitter thing DONE
#really should also have a max area DONE
"""
Author: Brian Brogan
Karolinska Intitutet
Fall 2024
This script is designed to create a controlled parameter experiment to determine results for various ranges of watershed segmentation.
"""

class ControlSegmenter():
    '''Performs controlled-variable watershed segmentation experiment on prerpocessed image.'''
    def __init__(self,
                 image_fname,
                 var_ranges: dict,
                 controls: list | None, # controls must be of length 4 -> [dtp, dks, minca, maxca] (even if not all are to be tested), or None
                 channel_id: int = 1, 
                 preproc_defaults: list = [1, (11,11), (5,5)] 
                 ):
        '''Constructor for ControlSegmenter class. Reads variable ranges and automatically runs segmentation experiment.
        Params:
            image_fname: (str) The filepath for the analysis image.
            var_ranges: (dict) Organizing dictionary of variables to be tested in segmentation. 
                        Options include distance transform percentile, dilation kernel size, 
                        minimum or maximum cell area. Any choice of these three variables is supported.
            controls: (list) The control list for the segmentation experiment. Controls must be 
                      of length 3 or None in the order of var_ranges even if not all variables are specified for 
                      testing. If controls is None, all 3 var_ranges must be specified and
                      the median of each var_range will be imputed.
            channel_id: (int) The channel of the image to be accessed.
            preproc_defaults: (list) Default settings for MaskMaker.preprocess(). Used in ... maybe also move this up'''
        
        assert controls is None or len(controls) == len(var_ranges), \
            "Controls must be None or a list of the same length (and order) as var_ranges.keys"
        assert var_ranges is not None and len(var_ranges) > 0, \
            "Test parameters and ranges must be specified for analysis."
        self.image_fname = image_fname
        self.channel_id = channel_id
        self.var_ranges = var_ranges
        self.var_ranges_keys = list(self.var_ranges.keys())
        self.var_ranges_values = list(self.var_ranges.values())
        self.controls = controls
        self.preproc_defaults = preproc_defaults
        self.preproc = self.preprocess(self.preproc_defaults[0], self.preproc_defaults[1], self.preproc_defaults[2])
        self.var_seg_fulldf = self.variable_segmentation_fulldf()

    def preprocess(self,
                   threshline=1,
                   gauss_ksize=(11,11), 
                   opening_ksize=(5,5)):
        '''Performs preprocessing steps for more accurate segmentation. Imports, slices, thresholds, blurs, and opens the 
        image specified at the image_fname attr.
        Params:
            threshline: (int) default = 0 The threshold of binarization of the image after Gaussian blurring.
            gauss_ksize: (tuple) default = (11,11) The kernel size for the Gaussian blurring.
            opening_ksize: (tuple) default = (5,5) The kernel size for the opening (erosion then dilation) applied to the image.
        Returns:
            opened_img: The single-channel preprocessed image array.'''
        
        assert self.image_fname is not None, "Image filepath must be specified to run analysis."
        img = cv2.imread(self.image_fname, cv2.IMREAD_UNCHANGED)
        if img is None:
            raise ValueError("Could not read the image file: {}".format(self.image_fname))
        if len(img.shape) > 2 and self.channel_id < img.shape[2]:
            img = img[:, :, self.channel_id]
        else:
            raise IndexError("Channel ID is out of bounds for the image dimensions.")
        print("Collected Image from ", self.image_fname, " with shape ", img.shape, flush=True)
        assert threshline < np.max(img), "Threshold above bounds of image intensity range."
        blur = cv2.GaussianBlur(img,
                                gauss_ksize, 0)
        _, bin_img = cv2.threshold(blur, threshline, np.max(img), cv2.THRESH_BINARY)
        opened_img = cv2.morphologyEx(bin_img, 
                                      cv2.MORPH_OPEN, 
                                      cv2.getStructuringElement(cv2.MORPH_ELLIPSE, opening_ksize), 
                                      iterations=1)
        return opened_img

    def variable_segmentation_fulldf(self):
        '''Method for populating the variable segmentation Dataframe. Implements the variable engine for each range of variables. 
        Combines controls with var_ranges iteratively. Called in ControlSegmenter initializer. 
        Params:
            None
        Returns:
            var_seg_fulldf: (pd.DataFrame) contains test_paramID, num_cells, dt_percentile, dilation_kernel_size, 
            minimum_cell_area, markers (arr), contours (arr).'''
        
        if self.controls is None:
            if len(self.var_ranges) != 3:
                raise ValueError("If full control list is not specified, var_ranges for all 3 params must be provided.")
            print("Imputing var_range medians for segmentation controls because none were specified.", flush=True)
            self.controls = [np.median(range).astype(np.uint8) for range in self.var_ranges_values]
        var_seg_fulldf = None
        for i in range(len(self.var_ranges_values)):
            print(f'Variable Engine: creating segmentations for {self.var_ranges_keys[i]} : {len(self.var_ranges_values[i])} settings', flush=True)
            test_params = self.controls.copy()
            test_params[i] = self.var_ranges_values[i]
            print(f'controls: {self.controls}', flush=True)
            print(f'var seg method: test_params: {test_params}', flush=True)
            idf = self.variable_engine(test_params)
            if var_seg_fulldf is None:
                var_seg_fulldf = idf
            else:
                var_seg_fulldf = pd.concat([var_seg_fulldf, idf], ignore_index=True)
        return var_seg_fulldf

    def variable_engine(self, params):
        '''The engine function for performing a round of variable watershed segmentation. Called in self.variable_segmentation_fulldf. 
        Calls self.watershed iterative for one range specified list element in params.
        Params:
            params: (list) List of control parameters containing one list of variables for testing at the desired test index.
        Returns:
            round: (pd.DataFrame) A sub DataFrame containing results from a single round of variable watershed segmentation. 
                   Intended to be concatenated to var_seg_fulldf iteratively in variable_segmentation_fulldf.'''
        
        round = pd.DataFrame(columns = ['test_paramID', 'num_cells'] + self.var_ranges_keys + ['markers', 'contours'])
        test_id = None
        for i, param in enumerate(params):
            control_type_check = isinstance(param, (list, np.ndarray))
            if control_type_check:
                test_id = i
                break
        if test_id is None:
            raise ValueError("Controls param must contain 1 list for iterative analysis.")
        for i, element in enumerate(params[test_id]):
            print(f'params testid: {params[test_id]}', flush=True)
            print(f'element: {element}')
            print(f'Engine running... testing setting {element}', flush=True)
            new_params = params.copy()
            new_params[test_id] = element
            markers, contour_arr = self.watershed(new_params)
            results = {'test_paramID': self.var_ranges_keys[test_id] + str(i)}
            results['num_cells'] = len(contour_arr)
            results.update({name: new_params[j] for j, name in enumerate(self.var_ranges_keys)})
            results['markers'] = markers.tolist() # keep as list for json
            results['contours'] = contour_arr
            round.loc[len(round)] = results
        return round

    def watershed(self, param_list):
        '''Verbosely performs a single watershed segmentation based on super.preproc using parameters in param_list.
        Params:
            param_list: (list) Contains the three test parameters in the following order: dtp, dks, minca, maxca.
        Returns: 
            markers: (array) Marker array calculated by openCV.watershed. Mimics shape of super.preproc.
            filtered_contours: (list) Sifted contour array calculated by self.findsift_contours.'''
        
        dtp,dks,minca,maxca = param_list
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3,3))
        steps = [
            ("dilating", lambda: cv2.dilate(self.preproc, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (dks, dks)), iterations=1)),
            ("distance transform", lambda: cv2.distanceTransform(self.preproc, cv2.DIST_L2, 5)),
            ("gaussian filter", lambda: gaussian_filter(dist, sigma=1)),
            ("percentile threshold", lambda: np.percentile(smoothed_dt, dtp)),
            ("connected components", lambda: cv2.connectedComponents(local_maxima)),
            ("watershed", lambda: cv2.watershed(cv2.cvtColor(self.preproc, cv2.COLOR_GRAY2BGR), np.int32(markers))),
        ]
        for step, operation in steps:
            result = operation()
            if step == 'dilating':
                sure_bg = cv2.dilate(self.preproc, kernel, iterations=3)
            if step == "distance transform":
                dist = result
                _, sure_fg = cv2.threshold(dist, 0.3 * dist.max(), 255, cv2.THRESH_BINARY)
                sure_fg = sure_fg.astype(np.uint8)
                unknown = cv2.subtract(sure_bg, sure_fg)
            elif step == "gaussian filter":
                smoothed_dt = result
            elif step == "percentile threshold":
                threshold = result
                local_maxima = (smoothed_dt > threshold).astype(np.uint8)
            elif step == "connected components":
                _, markers = result
                markers += 1
                markers[unknown == 255] = 0
            elif step == "watershed":
                markers = result

        print("Sifting contours...", flush=True)
        filtered_contours = self.findsift_contours(markers, minca, maxca)
        print(f'{len(filtered_contours)} CELLS FOUND', flush=True)
        return markers, filtered_contours

    def findsift_contours(self, markers, minca, maxca):
        '''Verbose contour finder for segmentation marker array. Creates binary mask for each marker and applies openCV.findContours().
          Once calculated contours are sifted for minimum/maximum area (minca, maxca).
        Params:
            markers: (array) Array of segmentation markers. Binary masks are created for each unqiue nonzero (background) value of the 
                     marker array
            minca: (int) The minimum required contour area for post-caluclation sifting.
            maxca: (int) The maximum allotted contour area for post-calculation sifitng.'''
        
        all_contours = []
        unique_labels = np.unique(markers)
        unique_labels = unique_labels[unique_labels != 0]

        with tqdm(total=len(unique_labels), desc='Sifting') as pbar:
            for i, label in enumerate(unique_labels, 1):
                binary_mask = (markers == label).astype(np.uint8) * 255
                contours, _ = cv2.findContours(binary_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
                filtered_contours = [cnt.tolist() for cnt in contours if maxca >= cv2.contourArea(cnt) >= minca]
                all_contours.extend(filtered_contours)
                pbar.set_postfix_str(f"{i}/{len(unique_labels)}")
                pbar.update(1)
        return all_contours

    def export(self, out_dir="mask_maker_output", parquet_name="variable_segmentation_metadata"):
        '''Export function for data collected by ControlSegmenter. Not automatically called during instance construction. 
        Ensures that DataFrame is correctly populated and exports parquet and json files. Parquet contains metadata from 
        variable segmenentation (test_paramID, num_cells, segmentation settings), markers.json and contours.json contains 
        the markers and contour arrays from each segmentation
        Params:
            out_dir (str) default = "mask_maker_output" The default name for output directory.
            parquet_name (str) default = "variable_segmentation_metadata" The default filename for metadata parquet.'''
        
        assert self.var_seg_fulldf is not None, \
        "Ensure that variable segmentation full Dataframe is initialized properly before attempting to report."
        os.makedirs(out_dir, exist_ok=True)
        parquet_path = os.path.join(out_dir, parquet_name + ".parquet")
        print("Creating exports...", flush=True)
        self.var_seg_fulldf.iloc[:, :-2].to_parquet(parquet_path, index=False)
        segmentation = self.var_seg_fulldf.iloc[:, -2:]
        for col in segmentation.columns: 
            json_path = os.path.join(out_dir, f"{col}.json")
            json_data = None
            if col == "markers":
                json_data = [arr for arr in segmentation[col]]
            elif col == "contours":
                json_data = segmentation[col].tolist()  
            assert json_data is not None, \
            "Ensure that markers and contours are properly created before attempting to report."
            with open(json_path, 'w') as f:
                json.dump(json_data, f)
        print("Done.", flush=True)

    



if __name__ == "__main__": 

    # Import Pathnames
    origin_csv_fname = None
    full_tiff = '/Users/brianbrogan/Desktop/KI24/ClusterImgGen2024/STERSEQ/output/02. Images/02.1. Raw prediction/1996-081_GFM_SS200000954BR_A2_tissue_cleaned_cortex_crop/clusters/cluster0_from_1996-081_GFM_SS200000954BR_A2_bin100_tissue_cleaned_mean_r10_cap1000.tif'
    tiff_slice = '/Users/brianbrogan/Desktop/KI24/figures/test_slice.tiff'

    image_fname = tiff_slice

    ## Variable Definition - start/stop - ensure range is divisible by step

    # Distance Transform Percentile for Watershed Marker Definition
    dtp = [80, 98]
    dtp_step = 1

    # Watershed Dilation Kernel Size
    dks = [2, 7]
    dks_step = 1

    # Minimum Cell Area Count (pixels)
    minca = [50, 200]
    minca_step = 25

    # Maximum Cell Area Count (pixels)
    maxca = [850, 1000]
    maxca_step = 25


    watershed_var_ranges = {
        'dt_percentile': np.arange(dtp[0], dtp[1]+1, dtp_step),
        'dilation_kernel_size': np.arange(dks[0], dks[1]+1, dks_step, dtype=np.uint8),
        'minimum_cell_area': np.arange(minca[0], minca[1]+1, minca_step),
        'maximum_cell_area': np.arange(maxca[0], maxca[1]+1, maxca_step)
    }

    print("Origin Data File: ", origin_csv_fname)
    print("Selected Image File: ", image_fname)
    print("Number of params to be analyzed: ", len(watershed_var_ranges))


    #tiff.imwrite('figures/MaskMaker_test1.tiff', masker.preproc, photometric='minisblack')
    test_segmenter = ControlSegmenter(image_fname= image_fname,
                                      var_ranges= watershed_var_ranges,
                                      controls= [90,
                                                 5,
                                                 100,
                                                 900],
                                      #area_filter_jobn=4,
                                      channel_id=1
                                      )
    test_segmenter.export()
    #test_segmenter.save_tiff_slice()
    print(test_segmenter.var_seg_fulldf.info())
    print(test_segmenter.var_seg_fulldf.head())
    #print(test_segmenter.var_seg_fulldf.shape)
    #print(len(test_segmenter.var_seg_fulldf.loc[0, 'contours']))