import os
import pandas as pd
import numpy as np
import cv2
from _io_data_generation import check_directory, find_movies, copy_movie
from LV_mask_analysis import Contour
import matplotlib.pyplot as plt
import networkx as nx
from sklearn.neighbors import NearestNeighbors
from scipy.spatial.distance import cdist
from itertools import combinations


class ExtractEdEs:

    def __init__(self, echonet_path=None, output_path=None):

        if echonet_path is not None:
            self.echonet_path = echonet_path
            self.movies_path = os.path.join(echonet_path, 'GoodX2Y2')
            self.output_path = check_directory(os.path.join(echonet_path, 'Output'))

        if output_path is not None:
            self.output_path = check_directory(output_path)

        self.df_volume_tracings = None
        self.list_of_movie_files = None
        self.movie_name = None

    def _get_volume_tracings(self):
        self.df_volume_tracings = pd.read_excel(
            os.path.join(self.echonet_path, 'VolumeTracings.xlsx'),
            index_col='FileName',
            sheet_name='VolumeTracings')
        # TESTING
        # self.df_volume_tracings = pd.read_excel(
        #     os.path.join(r'G:\DataGeneration\echonet_labels', 'VolumeTracingsTest.xlsx'),
        #     index_col='FileName',
        #     sheet_name='Sheet1')

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

    def _fix_contour(self, df_split_contour, plot_contour=False):

        def _remove_basal_points(_df, label='X1'):
            new_df = _df.copy()
            points = new_df[label].values
            dists = np.abs(np.diff(points))
            if dists[-1] > 3 * np.mean(dists):
                new_df = new_df.iloc[:-1]
                if dists[-2] > 3 * np.mean(dists):
                    new_df = new_df.iloc[:-1]
            return new_df

        df_1 = df_split_contour[['X1', 'Y1']].copy()
        df_1 = _remove_basal_points(df_1, 'X1')
        apex = df_1.iloc[0]
        df_2 = df_split_contour[['X2', 'Y2']].copy().iloc[1:]
        df_2 = _remove_basal_points(df_2, 'X2')
        df_2 = df_2.iloc[::-1]

        x = np.concatenate((df_2['X2'], df_1['X1']))
        y = np.concatenate((df_2['Y2'], df_1['Y1']))
        contour = np.array((x, y)).T

        # plt.plot(contour[:, 0], contour[:, 1], '.-')
        # plt.show()

        fixed_contour = self.sort_points_echonet_contour(contour, apex, False)

        if plot_contour:
            plt.plot(contour[:, 0], contour[:, 1], ':', label='contour')
            plt.plot(fixed_contour[:, 0], fixed_contour[:, 1], '-or', label='contour')
            plt.scatter(x=apex[0], y=apex[1],
                        c='b', marker='d', s=80, label='apex')
            plt.scatter(fixed_contour[0, 0], fixed_contour[0, 1], c='g', marker='d', s=80, label='left_basal')
            plt.scatter(fixed_contour[-1, 0], fixed_contour[-1, 1], c='k', marker='d', s=80, label='right_basal')
            plt.legend()
            plt.show()

        return fixed_contour, np.where(apex)[0][0]

    def sort_points_echonet_contour(self, points, _apex, show):

        perimeters, areas = [], []
        for i in range(1, 5):
            tri = np.array([points[0], _apex, points[-i]])
            perimeters.append(self._tri_len(tri))
            areas.append(self._get_contour_area(tri))

        score = np.array(perimeters) * np.array(areas)

        if np.argmax(score) == 0:
            new_points = points
        else:
            new_points = points[:-(np.argmax(score)), :]

        new_points = np.flipud(new_points)

        if show:
            xx = new_points[:, 0]
            yy = new_points[:, 1]
            plt.figure()
            plt.plot(xx, yy, 'd-')
            plt.scatter(new_points[-1, 0], new_points[-1, 1], c='r', s=70)
            plt.scatter(new_points[0, 0], new_points[0, 1], c='g', s=70)
            plt.scatter(_apex[0], _apex[1], c='k', s=70)
            plt.show()

        return new_points

    def sort_points_full_contour(self, points, show):

        def _sort_w_neighbours(_points, point_id=10):
            print('NearestNeighbors')
            clf = NearestNeighbors(2, n_jobs=-1).fit(_points)
            G = clf.kneighbors_graph()
            point_set = nx.from_scipy_sparse_matrix(G)
            opt_order = list(nx.dfs_preorder_nodes(point_set, point_id))
            _sorted_points = np.array([_points[new_id] for new_id in opt_order])
            return _sorted_points

        def _update_marker_ids(_points, _markers):
            _markers['id_left_basal'] = int(np.where(_markers['left_basal'] == _points)[0][0])
            _markers['id_right_basal'] = int(np.where(_markers['right_basal'] == _points)[0][0])
            _markers['id_apex'] = int(np.where(_markers['apex'] == _points)[0][0])
            return _markers

        def _get_corner_points(_points):
            distances = cdist(points, points)
            corner_points = np.argmax(distances, axis=0)
            unique, counts = np.unique(corner_points, return_counts=True)
            pareto_points = points[unique]
            print(pareto_points)
            combs = list(combinations(pareto_points, r=3))
            perimeters, areas, tris = [], [], []
            for tri in combs:
                tris.append(np.array(tri))
                perimeters.append(self._tri_len(np.array(tri)))
                areas.append(self._get_contour_area(np.array(tri)))
            score = np.array(perimeters) * np.array(areas)
            optimal_triangle = np.array(combs[int(np.argmax(score))])
            _markers = dict()
            basal_points = sorted(optimal_triangle, key=lambda x: (x[1]), reverse=True)[:2]
            _markers['left_basal'], _markers['right_basal'] = sorted(basal_points, key=lambda x: (x[0]))
            _markers['apex'] = sorted(optimal_triangle, key=lambda x: (x[1]), reverse=False)[0]
            _markers = _update_marker_ids(_points, _markers)
            return _markers

        points = _sort_w_neighbours(points)
        markers = _get_corner_points(points)
        points = _sort_w_neighbours(points, markers['id_left_basal'])
        markers = _update_marker_ids(points, markers)

        if markers['id_apex'] > markers['id_right_basal']:
            print('basal_direction')
            sorted_points = np.concatenate((points[0].reshape(1, -1), points[-1:markers['id_right_basal']-1:-1]))
            sorted_points = _sort_w_neighbours(sorted_points, markers['id_left_basal'])
            markers = _update_marker_ids(points, markers)
        else:
            print('apical direction')
            sorted_points = points[:markers['id_right_basal']+1]

        if show:
            xx = sorted_points[:, 0]
            yy = sorted_points[:, 1]
            plt.figure()
            plt.plot(xx, yy, 'd-')
            plt.scatter(markers['left_basal'][0], markers['left_basal'][1], c='r', s=70)
            plt.scatter(markers['right_basal'][0], markers['right_basal'][1], c='r', s=70)
            plt.scatter(markers['apex'][0], markers['apex'][1], c='r', s=70)
            plt.show()

        return sorted_points, markers

    def process_contours(self, movie_id, df_case_data, frame_numbers):
        contours = {'id': movie_id}
        phases = ['ed', 'es']
        for i, frame_number in enumerate(frame_numbers):
            df_contour = df_case_data.loc[df_case_data.Frame == frame_number]
            contour, apex_id = self._fix_contour(df_contour.copy())
            contour_area = self._get_contour_area(contour)
            contours[phases[i]] = {'contour': contour, 'contour_area': contour_area, 'frame': frame_number,
                                   'apex_id': apex_id}

        if contours['ed']['contour_area'] < contours['es']['contour_area']:
            contours['ed'], contours['es'] = contours['es'], contours['ed']

        return contours

    def process_movie(self, ed_frame, es_frame):
        dict_frames = {}
        vidcap = cv2.VideoCapture(os.path.join(self.movies_path, self.movie_name))
        success, _ = vidcap.read()
        vidcap.set(1, es_frame - 1)
        success, dict_frames['es'] = vidcap.read()
        vidcap.set(1, ed_frame - 1)
        success, dict_frames['ed'] = vidcap.read()

        return dict_frames

    def _save_contours(self, dict_contours):
        contours_path = check_directory(os.path.join(self.output_path, 'Contours'))
        np.savetxt(os.path.join(contours_path, '{}_ed.csv'.format(dict_contours['id'])),
                   dict_contours['ed']['contour'], fmt='%1.4f', delimiter=',')
        np.savetxt(os.path.join(contours_path, '{}_es.csv'.format(dict_contours['id'])),
                   dict_contours['es']['contour'], fmt='%1.4f', delimiter=',')

    def _save_screenshots(self, dict_contours):
        screenshot_path = check_directory(os.path.join(self.output_path, 'Phase_images'))

        default_im_size = 1024
        frame_images = self.process_movie(dict_contours['ed']['frame'], dict_contours['es']['frame'])
        for phase in ['ed', 'es']:
            orig_ed_height, orig_ed_width = frame_images[phase].shape[:2]
            drawing_contours = np.array([dict_contours[phase]['contour'][:, 0] * default_im_size / orig_ed_height,
                                         dict_contours[phase]['contour'][:, 1] * default_im_size / orig_ed_width]).T
            drawing_image = cv2.resize(frame_images[phase], (default_im_size, default_im_size))

            cv2.polylines(drawing_image, [np.int32(drawing_contours)], isClosed=False, color=(255, 0, 0), thickness=5)
            cv2.imwrite(os.path.join(screenshot_path, "{}_{}.jpg".format(dict_contours['id'], phase)), drawing_image)

    def _save_curvature_markers(self, dict_contours):

        curvature_indices_path = check_directory(os.path.join(self.output_path, 'Curvature_indices'))
        curvature_markers = []
        for phase in ('ed', 'es'):
            curvature_markers.append(dict_contours[phase]['curvature_markers'])
        df_curvature = pd.DataFrame(curvature_markers, index=['ed', 'es'])
        df_curvature.to_csv(os.path.join(curvature_indices_path, dict_contours['id'] + '_curv.csv'))

    def extract_case_data(self, save_contours=False, save_curvature_indices=True, save_screenshots=False):

        curvature_indices = None
        movie_id = os.path.splitext(os.path.basename(self.movie_name))[0]
        print('Case ID: {}'.format(movie_id))
        df_case = self.df_volume_tracings.loc[movie_id]
        frames = pd.unique(df_case['Frame'])
        assert len(frames) == 2, 'More than 2 contours found for case {}'.format(movie_id)
        contours = self.process_contours(movie_id, df_case, frames)
        cont = Contour(segmentations_path=None)
        for phase in ('ed', 'es'):
            cont.endo_sorted_edge, _ = cont._fit_border_through_pixels(contours[phase]['contour'])
            cont.curvature = cont._calculate_curvature()
            contours[phase]['curvature'] = cont.curvature
            contours[phase]['curvature_markers'] = cont._get_curvature_markers()

        if save_curvature_indices:
            print('Saving curvature indices, ID: {}'.format(contours['id']))
            self._save_curvature_markers(contours)

        if save_contours:
            print('Saving contours, ID: {}'.format(contours['id']))
            self._save_contours(contours)

        if save_screenshots:
            print('Saving phase images, ID: {}'.format(contours['id']))
            self._save_screenshots(contours)

        return curvature_indices

    def sort_movies(self):
        # good_x2y2_path = check_directory(os.path.join(self.echonet_path, 'GoodX2Y2'))
        # bad_x2y2_path = check_directory(os.path.join(self.echonet_path, 'BadX2Y2'))
        movie_id = os.path.splitext(os.path.basename(self.movie_name))[0]
        print('Case ID: {}'.format(movie_id))
        df_case = self.df_volume_tracings.loc[movie_id]
        frames = pd.unique(df_case['Frame'])
        assert len(frames) == 2, 'More than 2 contours found for case {}'.format(movie_id)

        for i, frame_number in enumerate(frames):
            df_contour = df_case.loc[df_case.Frame == frame_number]

            x = 0
            if df_contour['Y1'][0] > np.min(df_contour['Y2']):
                input('Press enter')
                plt.plot(df_contour['X1'], df_contour['Y1'], '.-')
                plt.plot(df_contour['X2'], df_contour['Y2'], '.-')
                plt.scatter(df_contour['X1'][0], df_contour['Y1'][0], c='r', marker='o')
                plt.scatter(df_contour['X1'][-1], df_contour['Y1'][-1], c='g', marker='d')
                plt.show()

            print(x)
            # points2 = df_contour['X1'].values
            # dists = np.abs(np.diff(points2))
            # print(dists)
            # print(np.mean(dists))
            # print(np.max(dists))
            #
        #
        #     if dists[0] != np.max(dists):
        #         good_contour = False
        #
        # if good_contour:
        #     copy_movie(good_x2y2_path, os.path.join(self.movies_path, self.movie_name))
        # else:
        #     copy_movie(bad_x2y2_path, os.path.join(self.movies_path, self.movie_name))

    def process_echonet(self):
        print('Reading volume tracings')
        self._get_volume_tracings()
        print('Volume tracings read')
        movies = find_movies(self.movies_path, 'avi')
        with open(os.path.join(self.output_path, 'Failed.txt'), 'w') as f:
            for movie in movies:
                self.movie_name = movie
                try:
                    self.extract_case_data(save_contours=True, save_curvature_indices=True, save_screenshots=True)
                    # self.sort_movies()
                except KeyError:
                    print('Key error')
                    input('Press enter')
                    f.write('Key error. Contour not found for case {}\n'.format(self.movie_name))
                except ValueError:
                    print('ValueError')
                    f.write('Value error. Contour calculation failed for case {}\n'.format(self.movie_name))
                except IndexError:
                    print('IndexError')
                    f.write('Index Error. Marker calculation failed for case {}\n'.format(self.movie_name))


if __name__ == '__main__':

    enet_path = 'G:\DataGeneration\EchoNet-Dynamic'  # contains 'echonet_labels' and 'Movies'
    ex = ExtractEdEs(enet_path)
    ex.process_echonet()

