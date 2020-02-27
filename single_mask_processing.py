import os
import numpy as np
import imageio
import matplotlib.pyplot as plt
from scipy.spatial.distance import cdist
from itertools import combinations, product
from LV_mask_analysis import Contour
from skimage import morphology, measure
from skimage import img_as_float


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
        - a dictionary with 4 keys:
            - contour = (500 points, smoothed, ordered from left upper point ['basal septal point'])
            - convexity - convexity metric value calculated as Area(mask) / Area(ConvexHull(mask))
            - simplicity - simplicity metric value calculated as (Sqrt(4 * PI * Area(mask)) / Perimeter(mask)
            - curvature_markers = dictionary with 6 keys, each containing the average value of the segment:
             'basal_curvature_1_mean_endo'
             'mid_curvature_1_mean_endo'
             'apical_curvature_1_mean_endo'
             'apical_curvature_2_mean_endo'
             'mid_curvature_2_mean_endo'
             'basal_curvature_2_mean_endo'
    """
    def __init__(self, mask=np.zeros((256, 256)), mask_value=1):
        self.mask = mask
        self.mask_value = mask_value
        self.mask[self.mask > 0] = mask_value
        self.sorted_edge_points = None
        self.sorted_endo_contour = None

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

    # -----EndocardialBorderSearch--------------------------------------------------------------------------------------
    def get_contour(self, show=False):
        _contour = measure.find_contours(self.mask, level=np.max(self.mask)/2)[0]
        self.sorted_edge_points = np.roll(_contour, 1, axis=1)
        self.sorted_edge_points = self.sorted_edge_points[:-1]
        if show:
            plt.imshow(self.mask, cmap='gray')
            plt.plot(self.sorted_edge_points[:, 0], self.sorted_edge_points[:, 1], 'r.', label='edge points')
            plt.plot(self.sorted_edge_points[:, 0], self.sorted_edge_points[:, 1], c='orange', label='contour')
            plt.legend()
            plt.show()
        return self.sorted_edge_points

    def get_endo_contour(self, show=False):

        def divide_pareto_points(pareto_points):
            centroid = np.mean(self.sorted_edge_points, axis=0)
            _basals = [p for p in pareto_points if p[1] > centroid[1]]
            _apicals = [p for p in pareto_points if p[1] < centroid[1]]
            return _basals, _apicals

        def find_optimal_base(_basals, _apicals):
            combs = combinations(_basals, r=2)
            prods = product(combs, _apicals)
            perimeters, areas, bases = [], [], []
            for tri in prods:
                base = [tri[0][0], tri[0][1]]
                bases.append(np.array(base))
                tri = np.array([tri[0][0], tri[0][1], tri[1]])
                perimeters.append(self._tri_len(np.array(tri)))
                areas.append(self._get_contour_area(np.array(tri)))

            score = np.array(perimeters) * np.array(areas)
            return np.array(bases[int(np.argmax(score))])

        self.get_contour(show)
        distances = cdist(self.sorted_edge_points, self.sorted_edge_points)
        corner_points = np.argmax(distances, axis=0)
        unique, counts = np.unique(corner_points, return_counts=True)
        pareto_points = self.sorted_edge_points[unique]

        basals, apicals = divide_pareto_points(pareto_points)
        optimal_base = find_optimal_base(basals, apicals)

        left_basal, right_basal = sorted(optimal_base, key=lambda x: (x[0]))
        left_basal_id = np.where((self.sorted_edge_points == left_basal).all(axis=1))[0]
        self.sorted_endo_contour = np.roll(self.sorted_edge_points, -left_basal_id, axis=0)
        right_basal_id = np.where((self.sorted_endo_contour == right_basal).all(axis=1))[0]
        self.sorted_endo_contour = self.sorted_endo_contour[:int(right_basal_id)]

        if show:
            plt.plot(self.sorted_endo_contour[:, 0], self.sorted_endo_contour[:, 1], 'r-')
            plt.plot(pareto_points[:, 0], pareto_points[:, 1], 'bo')
            plt.show()

        return self.sorted_endo_contour
    # ---END-EndocardialBorderSearch------------------------------------------------------------------------------------

    # -----ExecMethods--------------------------------------------------------------------------------------------------
    def get_simplicity(self):
        mask_area = np.sum(self.mask)
        mask_perimeter = measure.perimeter(self.mask)
        return (np.sqrt(np.pi*4*mask_area))/mask_perimeter

    def get_convexity(self, show=False):
        convex_hull = morphology.convex_hull_image(self.mask)
        mask_area = np.sum(self.mask)
        convex_hull_area = np.sum(convex_hull)

        if show:
            plt.subplot(221)
            plt.imshow(self.mask, cmap='gray')
            plt.title('Original mask')
            plt.subplot(222)
            plt.imshow(convex_hull, cmap='gray')
            plt.title('Convex hull')
            plt.subplot(223)
            chull_diff = img_as_float(convex_hull.copy())
            chull_diff[self.mask > 0] = 2 * self.mask_value
            plt.imshow(chull_diff, cmap='hot')
            plt.title('Comparison')
            plt.subplot(224)
            plt.imshow(convex_hull - self.mask, cmap='gray')
            plt.title('Difference')
            plt.tight_layout()
            plt.show()
        return mask_area/convex_hull_area

    def get_curvature(self, show=False):
        if self.sorted_endo_contour is None:
            self.get_endo_contour(show)
        border = Contour(segmentations_path=None)
        border.endo_sorted_edge, _ = border.fit_border_through_pixels(self.sorted_endo_contour)
        border.curvature = border.calculate_curvature()
        curvature_markers = border.get_curvature_markers()

        if show:
            plt.imshow(self.mask, cmap='gray')
            plt.plot(np.array(border.endo_sorted_edge)[:, 0], np.array(border.endo_sorted_edge)[:, 1], c='orange',
                     label='Smooth endo contour', linewidth=3)
            plt.plot(self.sorted_edge_points[:, 0], self.sorted_edge_points[:, 1], 'r.', label='Border points')
            plt.title('Smoothing results')
            plt.legend()
            plt.show()
        return curvature_markers

    def get_contour_and_markers(self, show=False):
        contour_markers = {'contour': self.get_endo_contour(show),
                           'curvature': self.get_curvature(show),
                           'simplicity': self.get_simplicity(),
                           'convexity': self.get_convexity(show)}
        return contour_markers
    # ---END-ExecMethods------------------------------------------------------------------------------------------------


if __name__ == '__main__':

    mask_path = r'G:\DataGeneration\Masks'
    mask_images = os.listdir(mask_path)
    for mask_image in mask_images:
        print(mask_image)
        _mask = imageio.imread(os.path.join(mask_path, mask_image))
        m2c = Mask2Contour(_mask)
        c_m = m2c.get_contour_and_markers(True)
        print(c_m['convexity'])
        print(c_m['simplicity'])
        print(c_m['curvature'])
