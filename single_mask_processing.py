import os
import numpy as np
import imageio
import matplotlib.pyplot as plt
from scipy.spatial import Voronoi, Delaunay
from scipy.spatial.distance import cdist
from itertools import combinations, product
from sklearn.neighbors import NearestNeighbors
import networkx as nx
from LV_mask_analysis import Contour


class Mask2Contour:
    """
    Class extracting the endocardial contour and its curvature indexes from a mask with LV bloodpool.
    Assumptions:
        - the value of mask is 255 and so is the maximum pixel value (as it is in a typical grayscale image). If it's
        not the case, an additional method should be called beforehand, to enforce it.
        - the mask is positioned in the way that the base is directed upwards, with septum on the left side
    Function to execute:
        - Mask2Contour.get_contour_and_curvature(self, show=False)
    Returns:
        - a dictionary with two keys:
            - contour = (500 points, smoothed, ordered from left upper point ['basal septal point'])
            - curvature_markers = dictionary with 6 keys, each containing the average value of the segment:
             'basal_curvature_1_mean_endo'
             'mid_curvature_1_mean_endo'
             'apical_curvature_1_mean_endo'
             'apical_curvature_2_mean_endo'
             'mid_curvature_2_mean_endo'
             'basal_curvature_2_mean_endo'
    """
    def __init__(self, mask=np.zeros((256, 256)), mask_value=255):
        self.mask = mask
        self.mask_value = mask_value
        self.edge_points = None
        self.sorted_edge_points = None
        self.edge = None

    @staticmethod
    def _pair_coordinates(edge):
        return np.array([(x, y) for x, y in zip(edge[0], edge[1])])

    @staticmethod
    def _get_contour_area(contour):
        x, y = contour[:, 0], contour[:, 1]
        return 0.5 * np.abs(np.dot(x, np.roll(y, 1)) - np.dot(y, np.roll(x, 1)))

    @staticmethod
    def _tri_len(triplet):
        triplet_shift = triplet.copy()
        triplet_shift = np.roll(triplet_shift, 1)
        perimeter = np.sum([np.linalg.norm(a - b) for a, b in zip(triplet, triplet_shift)])
        return perimeter

    # -----BorderExtraction---------------------------------------------------------------------------------------------
    def _correct_indices(self):
        # To make sure that endocardium and not the cavity is captured, relevant indices are moved by 1
        row_diffs = np.diff(self.mask, axis=1)
        row_diffs_right = list(np.where(row_diffs == self.mask_value))
        row_diffs_left = list(np.where(row_diffs == 256 - self.mask_value))
        col_diffs = np.diff(self.mask, axis=0)
        col_diffs_down = list(np.where(col_diffs == self.mask_value))
        col_diffs_up = list(np.where(col_diffs == 256 - self.mask_value))

        # index correction
        row_diffs_left = [row_diffs_left[0], row_diffs_left[1] + 0.5]
        col_diffs_up = [col_diffs_up[0] + 0.5, col_diffs_up[1]]
        row_diffs_right = [row_diffs_right[0], row_diffs_right[1] + 0.5]
        col_diffs_down = [col_diffs_down[0] + 0.5, col_diffs_down[1]]

        # putting points together
        edge = list()
        edge_y = np.concatenate((row_diffs_right[0], row_diffs_left[0], col_diffs_down[0], col_diffs_up[0]))
        edge_x = np.concatenate((row_diffs_right[1], row_diffs_left[1], col_diffs_down[1], col_diffs_up[1]))
        edge.append(edge_x)
        edge.append(edge_y)

        return self._pair_coordinates(edge)
    # ---END-BorderExtraction-------------------------------------------------------------------------------------------

    # -----SortingAlgorihtms--------------------------------------------------------------------------------------------
    def crust_algorithm(self, point_id=None, show=False):
        vor = Voronoi(self.edge_points)
        list_edge_points = self.edge_points.tolist()
        crust_set = np.unique(np.concatenate((self.edge_points, vor.vertices)), axis=0)
        deltri = Delaunay(crust_set)

        def simplices2edges(_deltri, _crust_set):
            unique_edges = {'points': [], 'indices': []}
            for simplex in _deltri.simplices:
                del_edges_i = combinations(simplex, 2)
                for _del_edge_i in del_edges_i:
                    _del_edge_i = np.sort(_del_edge_i)
                    _del_edge = _crust_set[_del_edge_i, :]
                    unique_edges['points'].append(_del_edge)
                    unique_edges['indices'].append(_del_edge_i)
            unique_edges['points'] = np.unique(unique_edges['points'], axis=0)
            unique_edges['indices'] = np.unique(unique_edges['indices'], axis=0)
            return unique_edges

        all_edges = simplices2edges(deltri, crust_set)
        crust_edges = {'points': [], 'indices': []}

        for del_edge, edge_id in zip(all_edges['points'], all_edges['indices']):
            if del_edge[0].tolist() in list_edge_points and del_edge[1].tolist() in list_edge_points:
                crust_edges['points'].append(del_edge)
                crust_edges['indices'].append(tuple(edge_id))

        G = nx.Graph()
        G.add_nodes_from([tuple(point) for point in crust_set])
        G.add_edges_from(crust_edges['indices'])
        opt_order = list(nx.dfs_preorder_nodes(G.nodes(), point_id))  # left to right. Better method to be found
        opt_order = [node for node in opt_order if not isinstance(node, tuple)]
        _sorted_points = np.array([crust_set[new_id] for new_id in opt_order])

        if show:
            plt.subplot(121)
            plt.plot(self.edge_points[:, 0], self.edge_points[:, 1], 'r.')
            for crust_edge in crust_edges['points']:
                plt.plot(crust_edge[:, 0], crust_edge[:, 1], 'b')
            plt.title('Found edges')

            plt.subplot(122)
            plt.title('Crust algorithm result')
            plt.plot(_sorted_points[:, 0], _sorted_points[:, 1], 'b')
            plt.plot(self.edge_points[:, 0], self.edge_points[:, 1], 'r.')
            plt.scatter(_sorted_points[0][0], _sorted_points[0][1], c='g', marker='d')
            plt.scatter(_sorted_points[-1][0], _sorted_points[-1][1], c='k', marker='d')
            plt.show()

        return _sorted_points

    def sort_w_neighbours(self, show=False, point_id=0):
        clf = NearestNeighbors(2, n_jobs=-1).fit(self.edge_points)
        G = clf.kneighbors_graph()
        point_set = nx.from_scipy_sparse_matrix(G)
        opt_order = list(nx.dfs_preorder_nodes(point_set, point_id))
        _sorted_points = np.array([self.edge_points[new_id] for new_id in opt_order])

        if show:
            plt.title('Sorted with nearest neighbors')
            plt.plot(_sorted_points[:, 0], _sorted_points[:, 1], 'b')
            plt.plot(self.edge_points[:, 0], self.edge_points[:, 1], 'r.')
            plt.scatter(_sorted_points[0][0], _sorted_points[0][1], c='g', marker='d')
            plt.scatter(_sorted_points[-1][0], _sorted_points[-1][1], c='k', marker='d')
            plt.show()

        return _sorted_points

    # ---END-SortingAlgorithms------------------------------------------------------------------------------------------

    # -----EndocardialBorderSearch--------------------------------------------------------------------------------------
    def _update_marker_ids(self, _markers):
        _markers['id_left_basal'] = int(np.where((self.sorted_edge_points == _markers['left_basal']).all(axis=1))[0])
        _markers['id_right_basal'] = int(np.where((self.sorted_edge_points == _markers['right_basal']).all(axis=1))[0])
        _markers['id_apex'] = int(np.where((self.sorted_edge_points == _markers['apex']).all(axis=1))[0])
        return _markers

    def _get_corner_points(self, show=False):
        distances = cdist(self.sorted_edge_points, self.sorted_edge_points)
        corner_points = np.argmax(distances, axis=0)
        unique, counts = np.unique(corner_points, return_counts=True)
        pareto_points = self.sorted_edge_points[unique]

        centroid = np.mean(self.sorted_edge_points, axis=0)
        basal_points = []
        apicals = []

        for u in unique:
            if self.sorted_edge_points[u, 1] > centroid[1]:
                basal_points.append(self.sorted_edge_points[u])
            else:
                apicals.append(self.sorted_edge_points[u])

        combs = combinations(basal_points, r=2)
        prods = product(combs, apicals)
        perimeters, areas, tris = [], [], []
        for tri in prods:
            tri = np.array([tri[0][0], tri[0][1], tri[1]])
            tris.append(np.array(tri))
            perimeters.append(self._tri_len(np.array(tri)))
            areas.append(self._get_contour_area(np.array(tri)))

        score = np.array(perimeters) * np.array(areas)
        optimal_triangle = np.array(tris[int(np.argmax(score))])
        _markers = dict()
        basal_points = sorted(optimal_triangle, key=lambda x: (x[1]), reverse=True)[:2]
        _markers['left_basal'], _markers['right_basal'] = sorted(basal_points, key=lambda x: (x[0]))
        _markers['apex'] = sorted(optimal_triangle, key=lambda x: (x[1]), reverse=False)[0]
        _markers = self._update_marker_ids(_markers)

        if show:
            plt.title('Pareto points')
            plt.plot(self.sorted_edge_points[:, 0], self.sorted_edge_points[:, 1], 'r-')
            plt.plot(pareto_points[:, 0], pareto_points[:, 1], 'bo')
            plt.show()

        return _markers

    # ---END-EndocardialBorderSearch------------------------------------------------------------------------------------

    # -----ExecMethods--------------------------------------------------------------------------------------------------
    def get_contour(self, show=False):
        self.edge_points = self._correct_indices()
        if show:
            plt.imshow(self.mask, cmap='gray')
            plt.plot(self.edge_points[:, 0], self.edge_points[:, 1], 'r.')
            plt.title('Edge points')
            plt.show()
        return self.edge_points

    def sort_contour(self, method='neighbors', show=False):
        if method == 'neighbors':
            self.sorted_edge_points = self.sort_w_neighbours(show=show)
        elif method == 'crust':
            self.sorted_edge_points = self.crust_algorithm(show=show)
        else:
            self.sorted_edge_points = None
            exit('Only "neighbor" and "crust" methods available')

        markers = self._get_corner_points(show=show)

        # assert the points are ordered clockwise
        if markers['id_left_basal'] < markers['id_apex'] < markers['id_right_basal'] or \
                markers['id_right_basal'] < markers['id_left_basal'] < markers['id_apex'] or \
                markers['id_apex'] < markers['id_right_basal'] < markers['id_left_basal']:
            self.sorted_edge_points = self.sorted_edge_points[::-1, :]
            markers = self._update_marker_ids(markers)

        # roll so that the left basal is on the last position
        self.sorted_edge_points = np.roll(self.sorted_edge_points, -(markers['id_left_basal'] + 1), axis=0)
        markers = self._update_marker_ids(markers)

        # reverse
        if markers['id_left_basal'] == len(self.sorted_edge_points) - 1:
            self.sorted_edge_points = self.sorted_edge_points[::-1, :]
            markers = self._update_marker_ids(markers)

        # check order. If it's fucked, then whole contour is fucked
        assert markers['id_left_basal'] < markers['id_apex'] < markers['id_right_basal'], 'Wrong contour order!'

        self.sorted_edge_points = self.sorted_edge_points[:markers['id_right_basal']+1]  # remove mitral part of base

        if show:
            self.plot_contour_with_markers(markers)

        return markers

    def get_contour_and_curvature(self, show=False):
        contour_markers = dict()
        self.get_contour()
        markers = self.sort_contour(method='neighbors', show=show)
        border = Contour(segmentations_path=None)
        border.endo_sorted_edge, _ = border._fit_border_through_pixels(self.sorted_edge_points)
        border.curvature = border._calculate_curvature()
        contour_markers['contour'] = np.array(border.endo_sorted_edge)
        contour_markers['curvature_markers'] = border._get_curvature_markers()
        if show:
            self.plot_contour_with_smoothing(markers, contour_markers['contour'])
        return contour_markers
    # ---END-ExecMethods------------------------------------------------------------------------------------------------

    # -----Plotting-----------------------------------------------------------------------------------------------------
    def plot_contour_with_markers(self, _markers):
        plt.imshow(self.mask, cmap='gray')
        plt.plot(self.edge_points[:, 0], self.edge_points[:, 1], 'b.', label='edge', markersize=1)
        plt.plot(self.sorted_edge_points[:, 0], self.sorted_edge_points[:, 1], 'r-', label='contour')
        plt.scatter(_markers['left_basal'][0], _markers['left_basal'][1], marker='d', c='g', label='left_basal')
        plt.scatter(_markers['right_basal'][0], _markers['right_basal'][1], marker='d', c='magenta', label='right_basal')
        plt.scatter(_markers['apex'][0], _markers['apex'][1], marker='d', c='orange', label='apex')
        plt.legend()
        plt.show()
        plt.close()

    def plot_contour_with_smoothing(self, _markers, _smoothed_border):
        plt.plot(_smoothed_border[:, 0], _smoothed_border[:, 1], c='orange')
        plt.imshow(self.mask, cmap='gray')
        plt.plot(self.sorted_edge_points[:, 0], self.sorted_edge_points[:, 1], 'r.', label='contour')
        plt.scatter(_markers['left_basal'][0], _markers['left_basal'][1], marker='d', c='g', label='left_basal')
        plt.scatter(_markers['right_basal'][0], _markers['right_basal'][1], marker='d', c='magenta',
                    label='right_basal')
        plt.scatter(_markers['apex'][0], _markers['apex'][1], marker='d', c='orange', label='apex')
        plt.legend()
        plt.show()
    # ---END-Plotting---------------------------------------------------------------------------------------------------


if __name__ == '__main__':

    mask_path = r'G:\DataGeneration\Masks'
    mask_images = os.listdir(mask_path)
    for mask_image in mask_images:
        _mask = imageio.imread(os.path.join(mask_path, mask_image))
        m2c = Mask2Contour(_mask)
        cont_marker = m2c.get_contour_and_curvature(show=False)  #
