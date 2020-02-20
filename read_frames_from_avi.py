import os
import pandas as pd
import numpy as np
import cv2
from _io_data_generation import check_directory, find_movies
import matplotlib.pyplot as plt


class ExtractEdEs:

    def __init__(self, echonet_path, output_ed_path=None, output_es_path=None):

        self.echonet_path = echonet_path
        if output_ed_path is not None:
            self.output_ed_path = check_directory(output_ed_path)
        else:
            self.output_ed_path = check_directory(os.path.join(echonet_path, 'ED'))

        if output_es_path is not None:
            self.output_es_path = check_directory(output_es_path)
        else:
            self.output_es_path = check_directory(os.path.join(echonet_path, 'ES'))

        self.movies_path = os.path.join(echonet_path, 'Movies')

        self.df_volume_tracings = None
        self.list_of_movie_files = None
        self.movie_name = None

    def _get_volume_tracings(self):
        self.df_volume_tracings = pd.read_excel(
            os.path.join(self.echonet_path, 'echonet_labels', 'VolumeTracingsTest.xlsx'),
            index_col='FileName',
            sheet_name='Sheet1')

    @staticmethod
    def _get_contour_area(contour):
        x = contour[:, 0]
        y = contour[:, 1]
        return 0.5 * np.abs(np.dot(x, np.roll(y, 1)) - np.dot(y, np.roll(x, 1)))

    @staticmethod
    def _fix_contour(df_split_contour, rel_basal_points=5, plot_contour=False):

        # Append contours 1 and 2 and remove artifacts
        df_split_contour.loc[:, ['X1', 'Y1']] = df_split_contour.loc[:, ['X1', 'Y1']].values[::-1, :]
        apex_id = df_split_contour.shape[0] - 1
        x = df_split_contour['X1'].append(df_split_contour['X2'][1:]).values
        y = df_split_contour['Y1'].append(df_split_contour['Y2'][1:]).values
        contour = np.array((x, y)).T

        # Find optimal left basal point
        def tri_len(triplet):
            triplet_shift = triplet.copy()
            triplet_shift.append(triplet_shift.pop(0))
            return np.sum([np.linalg.norm(a - b) for a, b in zip(triplet, triplet_shift)])

        peri = []
        for basal_left_point in contour[:rel_basal_points, :]:
            peri.append(tri_len([contour[-1, :], contour[apex_id, :], basal_left_point]))
        basal_point_id = np.argmax(peri)

        # Adjust the contour
        fixed_contour = contour[basal_point_id:, :]
        fixed_apex_id = apex_id - basal_point_id

        if plot_contour:
            plt.plot(contour[:, 0], contour[:, 1], ':', label='contour')
            plt.plot(fixed_contour[:, 0], fixed_contour[:, 1], '-or', label='contour')
            plt.scatter(x=fixed_contour[fixed_apex_id, 0], y=fixed_contour[fixed_apex_id, 1],
                        c='b', marker='d', s=80, label='apex')
            plt.scatter(fixed_contour[0, 0], fixed_contour[0, 1], c='g', marker='d', s=80, label='left_basal')
            plt.scatter(fixed_contour[-1, 0], fixed_contour[-1, 1], c='k', marker='d', s=80, label='right_basal')
            plt.legend()
            plt.show()

        return fixed_contour, fixed_apex_id

    def process_contours(self, movie_id, df_case_data, frame_numbers):
        contours = {'id': movie_id}
        phases = ['ed', 'es']
        for i, frame_number in enumerate(frame_numbers):
            df_contour = df_case_data.loc[df_case_data.Frame == frame_number]
            contour, apex_id = self._fix_contour(df_contour.copy(), plot_contour=False)
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
        ed_contours_path = check_directory(os.path.join(self.output_ed_path, 'Contours'))
        np.savetxt(os.path.join(ed_contours_path, '{}_ed.csv'.format(dict_contours['id'])),
                   dict_contours['ed']['contour'], fmt='%1.4f', delimiter=',')

        es_contours_path = check_directory(os.path.join(self.output_es_path, 'Contours'))
        np.savetxt(os.path.join(es_contours_path, '{}_es.csv'.format(dict_contours['id'])),
                   dict_contours['es']['contour'], fmt='%1.4f', delimiter=',')

    def _save_screenshots(self, dict_contours):
        default_im_size = 1024
        frame_images = self.process_movie(dict_contours['ed']['frame'], dict_contours['es']['frame'])

        for phase in ['ed', 'es']:
            orig_ed_height, orig_ed_width = frame_images[phase].shape[:2]
            drawing_contours = np.array([dict_contours[phase]['contour'][:, 0] * default_im_size / orig_ed_height,
                                         dict_contours[phase]['contour'][:, 1] * default_im_size / orig_ed_width]).T
            drawing_image = cv2.resize(frame_images[phase], (default_im_size, default_im_size))

            if phase == 'ed':
                screenshot_path = check_directory(os.path.join(self.output_ed_path, 'Phase_images'))
            else:
                screenshot_path = check_directory(os.path.join(self.output_es_path, 'Phase_images'))

            cv2.polylines(drawing_image, [np.int32(drawing_contours)], isClosed=False, color=(255, 0, 0), thickness=5)
            cv2.imwrite(os.path.join(screenshot_path, "{}_{}.jpg".format(dict_contours['id'], phase)), drawing_image)

    def extract_case_data(self, save_contours=False, save_curvature_indices=True, save_screenshots=False):

        curvature_indices = None
        movie_id = os.path.splitext(self.movie_name)[0]
        print('Case ID: {}'.format(movie_id))
        df_case = self.df_volume_tracings.loc[movie_id]
        frames = pd.unique(df_case['Frame'])
        assert len(frames) == 2, 'More than 2 contours found for case {}'.format(movie_id)
        contours = self.process_contours(movie_id, df_case, frames)
    
        if save_curvature_indices:

            # TODO: first of, try to combine with LV mask analysis
            # TODO: smoothing
            # TODO: curvature along the contour
            # TODO: divide into 6 segments
            # TODO: save all relevant data
            curvature_indices = None

        if save_contours:
            print('Saving contours, ID: {}'.format(contours['id']))
            self._save_contours(contours)

        if save_screenshots:
            print('Saving phase images, ID: {}'.format(contours['id']))
            self._save_screenshots(contours)

        return curvature_indices

    def process_echonet(self):
        print('Reading volume tracings')
        self._get_volume_tracings()
        print('Volume tracings read')
        self.extract_case_data(save_contours=False, save_curvature_indices=False, save_screenshots=True)


if __name__ == '__main__':

    enet_path = 'G:\DataGeneration'  # contains 'echonet_labels' and 'Movies'
    ex = ExtractEdEs(enet_path)
    ex.movie_name = '0X1A2A76BDB5B98BED.avi'
    ex.process_echonet()
    # ex.process_movie(r'G:\DataGeneration\Movies\0X1A2A76BDB5B98BED.avi')

