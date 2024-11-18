import os 
import sys
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import json
import fastparquet
import tqdm
from cv2 import drawContours, pointPolygonTest

class DataManager:
    def __init__(self,
                 origin_csv_fname: str = None,
                 annotation_fname: str = None, 
                 metadata_fname: str = "mask_maker_output/variable_segmentation_metadata.parquet",
                 markers_fname: str = "mask_maker_output/markers.json",
                 contours_fname: str = "mask_maker_output/contours.json",
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
    
    #shit
    def get_single_marker_points(self,
                          marker_array: np.ndarray, # single marker array
                          target: int # target marker
                          ):
        target_points = np.where(marker_array == target)
        #target_points = np.column_stack(np.where(marker_array == target))
        return target_points
    
    #shit
    def get_single_contour_points(self, 
                           contour: np.ndarray # single contour
                           ):
        contour = contour.reshape((-1, 2))
        min_x, min_y = np.min(contour, axis=0)
        max_x, max_y = np.max(contour, axis=0)
        grid_points = [(x, y) for x in range(min_x, max_x+1) for y in range(min_y, max_y+1)]
        points_with_test_results = [pointPolygonTest(contour, pt, False) for pt in grid_points]
        inside_points = [grid_points[i] for i, result in enumerate(points_with_test_results) if result >= 0]
        return np.array(inside_points)
    
    def create_indicator_mask(self,
                              #origin_img_fname: str,
                              xrange: tuple = None,
                              yrange: tuple = None,
                              indicators: dict = None #key is a string, value is 1 or -1 to show positive or negative inficator
                              ):
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
    


