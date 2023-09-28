import pandas as pd
import numpy as np

from sealsml import geometry

def add_geo(ref_array, target_array, pd_export=False, column_names=["distance", "azimuth_cos", "azimuth_sin", "elevation_angle"]):
    """
    Calculates the geometric metrics between two arrays of points.

    Args:
        ref_array (numpy.ndarray): A numpy array of points, with shape (n, 3). Usually the reference points.
        target_array (numpy.ndarray): A numpy array of points, with shape (n, 3). Usually the target points.
        pd_export (bool, optional): Whether to export the results as a Pandas DataFrame. Defaults to False.
        column_names (list[str], optional): The column names for the exported Pandas DataFrame. Defaults to ["distance", "azimuth_cos", "azimuth_sin", "elevation_angle"].

    Returns:
        numpy.ndarray or pd.DataFrame: The geometric metrics, with shape (n, 4). If pd_export is True, a Pandas DataFrame is returned.
    """
    geometry_class = geometry.geo(ref_array, target_array)
    
    ### Let's get these metrics! 
    # distance
    distance  = geometry_class.distance_between_points_3d()
    
    # azimuth
    azi = geometry_class.calculate_azimuth()
    azi_cos = np.cos(np.radians(azi))
    azi_sin = np.sin(np.radians(azi))
    # elevation angle
    ele_angle = geometry_class.calculate_elevation_angle()

    combined_array = np.column_stack((distance, azi_cos, azi_sin, ele_angle))
    if pd_export:
        return pd.DataFrame(combined_array, columns=column_names)
    else:
        return combined_array