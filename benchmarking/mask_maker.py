import numpy as np
import pandas as pd
import cv2 
from scipy.ndimage import gaussian_filter
import seaborn as sb
import tifffile as tiff
from multiprocessing import Pool, cpu_count
import sys
import progressbar
import requests 
import time
from tqdm import tqdm

import gc
from joblib import Parallel, delayed


#do print('.', end=' ', flush=True) for print statements, also add more print statements - DONE
#bit depth
#maybe move preprocess to ControlSegmenter - NO

# Watershed method is SUPER SLOW -> change for tiled computation - NO DONE

#Ensure that find_all_contours actually finds all contours - DONE
#Check that image data unchanged (depth)
#Implement reporting:
#   progressbar - DONE
#   one for each param being tested  - DONE
#   also create updating statememt for process being executed - DONE
#Debug other params - DONE
#Inpsect dataframe creation - DONE
# Still want to try to run multiprocess... self.area_filter_jobn
#Organize Dir for comparison blueprint
    # OMIT Suzuki Abe option
    # Dilation Kernel size doesnt do anything so figure that out
    # Find suitable DF export type
    # Blueprint outside class for comparison
# Clean up everything (reports, imports, exe code)
# Documentation!! (docstrings, github)



class MaskMaker:
    def __init__(self, 
                 image_fname,
                 origin_csv_fname=None,
                 channel_id=1):
        self.image_fname = image_fname
        self.origin_csv_fname = origin_csv_fname
        self.channel_id = channel_id
        self.preproc = None #self.preprocess()

    def preprocess(self,
                   threshline=0,
                   gauss_ksize=(11,11), 
                   opening_ksize=(5,5)):

        # Import image file
        assert self.image_fname is not None, "Image filepath must be specified to run analysis."
        img = cv2.imread(self.image_fname, cv2.IMREAD_UNCHANGED)
        if img is None:
            raise ValueError("Could not read the image file: {}".format(self.image_fname))
        if len(img.shape) > 2 and self.channel_id < img.shape[2]:
            img = img[:, :, self.channel_id]
        else:
            raise IndexError("Channel ID is out of bounds for the image dimensions.")
        #print("Collected Image from ", self.image_fname, " with shape ", img.shape, flush=True)
        # Blur and Threshold
        assert threshline < np.max(img), "Threshold above bounds of image intensity range."
        blur = cv2.GaussianBlur(img,
                                gauss_ksize, 0)
        _, bin_img = cv2.threshold(blur, threshline, np.max(img), cv2.THRESH_BINARY) # Change to be DYNAMIC - will be changed for segmentation_method = suzukiAbe
        # Open
        opened_img = cv2.morphologyEx(bin_img, 
                                      cv2.MORPH_OPEN, 
                                      cv2.getStructuringElement(cv2.MORPH_ELLIPSE, opening_ksize), 
                                      iterations=1)
        return opened_img
    

    def compare(self, var_seg_full_df):
        ### DIFFERENT CLASS
        # make segmenters
        # perform roc for each segmenter's out_masks against origin csv
        # plot metrics for each
        pass

class ControlSegmenter(MaskMaker):
    def __init__(self,
                 image_fname,
                 var_ranges: dict,
                 controls: list | None, # controls must be of length 3 -> [dtp, dks, mca] (even if not all are to be tested), or None
                 area_filter_jobn: int = 4,
                 origin_csv_fname=None,
                 channel_id=1,
                 segmentation_method='watershed',
                 preproc_defaults = [0, (11,11), (5,5)]
                 ):
        super().__init__(image_fname, origin_csv_fname, channel_id)
        assert controls is None or len(controls) == 3, \
            "Controls must be None or a list of the same length (and order) as var_ranges.keys"
        assert var_ranges is not None and len(var_ranges) > 0, \
            "Test parameters and ranges must be specified for analysis."
        self.var_ranges = var_ranges
        self.var_ranges_keys = list(self.var_ranges.keys())
        self.var_ranges_values = list(self.var_ranges.values())
        self.controls = controls
        self.area_filter_jobn = area_filter_jobn
        self.segmentation_method = segmentation_method
        self.preproc_defaults = preproc_defaults
        self.var_seg_fulldf = self.variable_segmentation_fulldf()

    def variable_segmentation_fulldf(self):
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
        out_masks = pd.DataFrame(columns = ['test_paramID', 'num_cells'] + self.var_ranges_keys + ['markers', 'contour array'])
        test_id = None
        for i, param in enumerate(params):
            #if not isinstance(param, np.array):
            if isinstance(param, (list, np.ndarray, tuple)):
            #if len(param) > 1:
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
            if self.segmentation_method == 'SuzukiAbe':
                markers, contour_arr = self.suzukiAbe(new_params)
            if self.segmentation_method == 'watershed':
                self.preproc = self.preprocess()
                markers, contour_arr = self.watershed(new_params)
            results = {'test_paramID': self.var_ranges_keys[test_id] + str(i)}
            results['num_cells'] = len(contour_arr)
            results.update({name: new_params[j] for j, name in enumerate(self.var_ranges_keys)})
            results['markers'] = np.array(markers)
            results['contour array'] = contour_arr
            out_masks.loc[len(out_masks)] = results
        return out_masks

    def watershed(self, param_list):
        dtp, dks, mca = param_list
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (int(dks), int(dks)))
        report = "Watershed Engine: {}"
        steps = [
            ("dilating", lambda: cv2.dilate(self.preproc, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (dks, dks)), iterations=1)),
            ("distance transform", lambda: cv2.distanceTransform(self.preproc, cv2.DIST_L2, 5)),
            ("gaussian filter", lambda: gaussian_filter(dist, sigma=1)),
            ("percentile threshold", lambda: np.percentile(smoothed_dt, dtp)),
            ("connected components", lambda: cv2.connectedComponents(local_maxima)),
            ("watershed", lambda: cv2.watershed(cv2.cvtColor(self.preproc, cv2.COLOR_GRAY2BGR), np.int32(markers))),
        ]
        for step, operation in steps:
            #print(report.format(step), flush=True)
            result = operation()
            if step == 'dilating':
                sure_bg = cv2.dilate(self.preproc, kernel, iterations=1)
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
        filtered_contours = self.find_all_contours(markers, mca)
        print(f'{len(filtered_contours)} CELLS FOUND', flush=True)
        return markers, filtered_contours

    def find_all_contours(self, markers, mca):
        all_contours = []
        unique_labels = np.unique(markers)
        unique_labels = unique_labels[unique_labels != 0]

        with tqdm(total=len(unique_labels), desc='Sifting') as pbar:
            for i, label in enumerate(unique_labels, 1):
                binary_mask = (markers == label).astype(np.uint8) * 255
                contours, _ = cv2.findContours(binary_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
                filtered_contours = [cnt for cnt in contours if cv2.contourArea(cnt) >= mca]
                all_contours.extend(filtered_contours)

                pbar.set_postfix_str(f"{i}/{len(unique_labels)}")
                pbar.update(1)

        return all_contours

    def suzukiAbe(self, param_list):
        # performs single openCV contour find thing, returns mask and contour array
        filtered_contours = None
        preproc_img = self.preprocess(param_list[0], param_list[1], param_list[2])
        contours, _ = cv2.findContours(preproc_img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
        if contours: # and cv2.contourArea(contours[0]) > mca:  # Check if contours are found
            filtered_contours = [cnt for cnt in contours if cv2.contourArea(cnt) > mca]
            #cont_arr.append(contours[0])
        return np.nan, filtered_contours


if __name__ == "__main__": 

    # Import Pathnames
    origin_csv_fname = None
    full_tiff = '/Users/brianbrogan/Desktop/KI24/ClusterImgGen2024/STERSEQ/output/02. Images/02.1. Raw prediction/1996-081_GFM_SS200000954BR_A2_tissue_cleaned_cortex_crop/clusters/cluster0_from_1996-081_GFM_SS200000954BR_A2_bin100_tissue_cleaned_mean_r10_cap1000.tif'
    png_slice = '/Users/brianbrogan/Desktop/KI24/figures/color_slice.png'
    segmentation_method = 'watershed' #'SuzukiAbe', 'watershed' 

    image_fname = png_slice

    ## Variable Definition - start/stop - ensure range is divisible by step
    # Distance Transform Percentile for Watershed Marker Definition
    dtp = [80, 98]
    dtp_step = 1

    # Watershed Dilation Kernel Size
    dks = [2, 7]
    dks_step = 1

    # Minimum Cell Area Count (pixels)
    mca = [50, 200]
    mca_step = 10


    watershed_var_ranges = {
        'dt_percentile': np.arange(dtp[0], dtp[1]+1, dtp_step),
        'dilation_kernel_size': np.arange(dks[0], dks[1]+1, dks_step, dtype=np.uint8),
        'minimum_cell_area': np.arange(mca[0], mca[1]+1, mca_step)
    }

    suzukiAbe_var_ranges = {} # threshline(percentage?), gauss blur k size, opening k size

    print("Origin Data File: ", origin_csv_fname)
    print("Selected Image File: ", image_fname)
    print("Number of params to be analyzed: ", len(watershed_var_ranges))
    print("Segmentation method: ", segmentation_method)


    
    #tiff.imwrite('figures/MaskMaker_test1.tiff', masker.preproc, photometric='minisblack')
    test_segmenter = ControlSegmenter(image_fname= image_fname,
                                      var_ranges= watershed_var_ranges,
                                      controls= [90, 3, 100],
                                      area_filter_jobn=4,
                                      origin_csv_fname=None,
                                      channel_id=1,
                                      segmentation_method='watershed')
    print(test_segmenter.var_seg_fulldf.info())
    print(test_segmenter.var_seg_fulldf)