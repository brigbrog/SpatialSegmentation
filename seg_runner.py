from benchmarking.control_segmenter import ControlSegmenter
from benchmarking.manager import DataManager
from benchmarking.comparator import Indicator, Comparator
import numpy as np

# Import Pathnames
origin_csv_fname = '/Users/brianbrogan/Desktop/KI24/ClusterImgGen2024/STERSEQ/input/1996-081_GFM_SS200000954BR_A2_tissue_cleaned_cortex_crop.csv'
annotation_fname = '/Users/brianbrogan/Desktop/KI24/ClusterImgGen2024/STERSEQ/output/01. Co-expression network/1996-081_GFM_SS200000954BR_A2_bin100_tissue_cleaned/Cluster_annotation.csv'
full_tiff = '/Users/brianbrogan/Desktop/KI24/ClusterImgGen2024/STERSEQ/output/02. Images/02.1. Raw prediction/1996-081_GFM_SS200000954BR_A2_tissue_cleaned_cortex_crop/clusters/cluster0_from_1996-081_GFM_SS200000954BR_A2_bin100_tissue_cleaned_mean_r10_cap1000.tif'
png_slice = '/Users/brianbrogan/Desktop/KI24/figures/color_slice.png'

image_fname = png_slice

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

# Aggregate Variable Ranges
var_ranges = {
        'dt_percentile': np.arange(dtp[0], dtp[1]+1, dtp_step),
        'dilation_kernel_size': np.arange(dks[0], dks[1]+1, dks_step, dtype=np.uint8),
        'minimum_cell_area': np.arange(minca[0], minca[1]+1, minca_step),
        'maximum_cell_area': np.arange(maxca[0], maxca[1]+1, maxca_step)
    }

controls = [90,5,100,900]


if __name__ == "__main__": 

    # Run Variable Segmentation
    segmenter = ControlSegmenter(
        image_fname = image_fname,
        channel_id = 1,
        var_ranges = var_ranges,
        controls = controls
    )
    segmenter.export()

    # Collect Output
    manager = DataManager(
        origin_csv_fname = origin_csv_fname,
        annotation_fname = annotation_fname
    )

    # Pass Output to Indicator, Comparator
    indicator = Indicator(
        origin_csv = manager.origin_csv,
        annotation = manager.annotation,
    )

    comparator = Comparator(
        indicator = indicator,
        metadata = manager.metadata,
        markers = manager.markers,
        contours = manager.contours,
        rep_perc = 1.0
    )

    
    


