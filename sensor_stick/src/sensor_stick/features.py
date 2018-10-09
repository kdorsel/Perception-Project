import matplotlib.colors
import matplotlib.pyplot as plt
import numpy as np
from pcl_helper import *


def rgb_to_hsv(rgb_list):
    rgb_normalized = [1.0*rgb_list[0]/255, 1.0*rgb_list[1]/255, 1.0*rgb_list[2]/255]
    hsv_normalized = matplotlib.colors.rgb_to_hsv([[rgb_normalized]])[0][0]
    return hsv_normalized


def compute_color_histograms(cloud, using_hsv=True):

    # Compute histograms for the clusters
    point_colors_list = np.empty((0,3), int)

    # Step through each point in the point cloud
    for point in pc2.read_points(cloud, skip_nans=True):
        rgb_list = float_to_rgb(point[3])
        if using_hsv:
            point_colors_list = np.append(point_colors_list, [rgb_to_hsv(rgb_list) * 255], axis=0)
        else:
            point_colors_list = np.append(point_colors_list, [rgb_list], axis=0)

    # TODO: Compute histograms
    h_hist = np.histogram(point_colors_list[:,0], bins=64, range=(0, 256))
    s_hist = np.histogram(point_colors_list[:,1], bins=64, range=(0, 256))
    v_hist = np.histogram(point_colors_list[:,2], bins=64, range=(0, 256))

    hist_features = np.concatenate((h_hist[0], s_hist[0], v_hist[0])).astype(np.float64)
    norm_features = hist_features / np.sum(hist_features)

    return norm_features


def compute_normal_histograms(normal_cloud):
    norm_vals = np.empty((0,3), int)
    for norm_component in pc2.read_points(normal_cloud,
                                          field_names = ('normal_x', 'normal_y', 'normal_z'),
                                          skip_nans=True):
        norm_vals = np.append(norm_vals, [norm_component], axis=0)

    # TODO: Compute histograms of normal values (just like with color)
    h_hist = np.histogram(norm_vals[:,0], bins=50, range=(-1,1))
    s_hist = np.histogram(norm_vals[:,1], bins=50, range=(-1,1))
    v_hist = np.histogram(norm_vals[:,2], bins=50, range=(-1,1))
    # TODO: Concatenate and normalize the histograms
    hist_features = np.concatenate((h_hist[0], s_hist[0], v_hist[0])).astype(np.float64)
    norm_features = hist_features / np.sum(hist_features)

    return norm_features
