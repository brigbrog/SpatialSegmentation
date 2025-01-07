import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import json
import fastparquet
import tqdm
from cv2 import drawContours, pointPolygonTest


# Manager Class for S.T. origin, annotation, meta, markers, and contours data. 
class DataManager:
    def __init__(self,
                 origin_csv_fname: str = None,
                 annotation_fname: str = None, 
                 metadata_fname: str = "/Users/brianbrogan/Desktop/KI24/mask_maker_output/variable_segmentation_metadata.parquet",
                 markers_fname: str = "/Users/brianbrogan/Desktop/KI24/mask_maker_output/markers.json",
                 contours_fname: str = "/Users/brianbrogan/Desktop/KI24/mask_maker_output/contours.json",
                 ):
        self.parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        self.origin_csv_fname = origin_csv_fname
        self.annotation_fname = annotation_fname
        self.metadata_fname = metadata_fname
        self.markers_fname = markers_fname
        self.contours_fname = contours_fname
        self.origin_csv = pd.read_csv(self.origin_csv_fname)
        self.annotation = pd.read_csv(self.annotation_fname)
        self.metadata = self.import_metadata()
        self.markers = self.import_markers() # Pandas series where each element is a 2D numpy array
        self.contours = self.import_contours() # Pandas series where each element is list of numpy arrays (like CV)

    def import_metadata(self):
        pq_path = os.path.join(self.parent_dir, self.metadata_fname)
        metadata = pd.read_parquet(pq_path, engine='fastparquet')
        return metadata
    
    def import_markers(self):
        markers_path = os.path.join(self.parent_dir, self.markers_fname)
        with open(markers_path, 'r') as f:
            markers_data = json.load(f)
        mark_series = pd.Series(markers_data, name="markers_arr")
        mark_series = mark_series.apply(lambda mark: np.array(mark, dtype=np.int32))
        return mark_series
    
    def import_contours(self):
        contours_path = os.path.join(self.parent_dir, self.contours_fname)
        with open(contours_path, 'r') as f:
            contours_data = json.load(f)
        cont_series = pd.Series(contours_data, name="contour_arrs")
        cont_series = cont_series.apply(
            lambda iseg_cont_list: [np.array(cont, dtype=np.int32) for cont in iseg_cont_list])
        return cont_series


    ## Visualizers ##

    def view_marker_array(self,
                          marker_array: np.ndarray
                          ):
        plt.imshow(marker_array, cmap="tab20b")
        plt.title("Marker Array Visualization")
        plt.axis("off")
        plt.show()

    def view_single_contour(self, 
                            contour_array: list,
                            target_id: int
                            ):
        target = contour_array[target_id]
        x_vals, y_vals = zip(*[point[0] for point in target])
        min_x, max_x = min(x_vals), max(x_vals)
        min_y, max_y = min(y_vals), max(y_vals)
        margin = 10
        min_x, max_x = min_x - margin, max_x + margin
        min_y, max_y = min_y - margin, max_y + margin
        adjusted_contour = target - np.array([[min_x, min_y]])
        image_width = max_x - min_x + 1
        image_height = max_y - min_y + 1
        image = np.zeros((image_height, image_width), dtype=np.uint8)
        drawContours(image, [adjusted_contour], -1, 255, thickness=1)
        plt.imshow(image, cmap='gray')
        plt.title("Contour Visualization")
        plt.axis("off")
        plt.show()

    def visualize_contours(self, 
                           contours):
        min_x, min_y = float('inf'), float('inf')
        max_x, max_y = -float('inf'), -float('inf')
        for i, contour in enumerate(contours):
            x_vals, y_vals = zip(*[point[0] for point in contour]) 
            min_x, min_y = min(min_x, min(x_vals)), min(min_y, min(y_vals))
            max_x, max_y = max(max_x, max(x_vals)), max(max_y, max(y_vals))
        image = np.zeros((max_y - min_y + 10, max_x - min_x + 10), dtype=np.uint8)
        for contour in contours:
            contour_shifted = np.array([[pt[0][0] - min_x, pt[0][1] - min_y] for pt in contour], dtype=np.int32)
            drawContours(image, [contour_shifted], -1, 255, thickness=1)
        plt.imshow(image, cmap='gray')
        plt.title("Contours Visualization")
        plt.axis('off')
        plt.show()
    