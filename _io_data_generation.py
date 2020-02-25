import os
import glob
import shutil


def check_directory(directory):
    # Check if directory exists; if not create it
    if not os.path.isdir(directory):
        os.mkdir(directory)
    return directory


def find_movies(directory, ext):
    movies = glob.glob(os.path.join(directory, '*.' + ext))
    return movies


def copy_movie(directory, movie_path):
    shutil.copy(movie_path, directory)
