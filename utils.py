# --------------------------------------------------------
# Duplicate Detection
# Licensed under The GPL2 License [see LICENSE for details]
# Written by Matthes Krull
# --------------------------------------------------------

from numba import njit
import numpy as np
import time
import cv2
from config import get_config
import argparse
import os


def parse_args():
    """
        Merges the command line configs with the ones form the config file
    Returns:
        args: Args from command line (not really needed)
        config: Yacs config file that is merged with args
    """
    parser = argparse.ArgumentParser(
        "Delete similar images in folder with cv2", add_help=False
    )

    # easy config modification
    parser.add_argument(
        "--path-dataset",
        type=str,
        required=False,
        default=None,
        metavar="PATH",
        help="path to dataset",
    )
    parser.add_argument(
        "--cfg",
        type=str,
        required=False,
        metavar="FILE",
        help="path to config file",
    )
    parser.add_argument(
        "--opts",
        help="Modify config options by adding 'KEY VALUE' pairs. ",
        default=None,
        nargs="+",
    )

    args, unparsed = parser.parse_known_args()

    config = get_config(args)

    return args, config


@njit
def pairwise_combs_numba(array):
    """
        https://localcoder.org/fast-combinations-without-replacement-for-arrays-numpy-python
        Computes all permutations between a & a, without repetitions, order not important.
        10x faster than "itertools.combinations"
    Args:
        array (Sequence): The sequence which is permutated

    Returns:
        list: Numpy matrix with shape (n,2) (all permutations)
    """

    n = len(array)
    L = n * (n - 1) // 2
    L = n * (n - 1) // 2
    out = np.empty((L, 2), dtype=array.dtype)
    iterID = 0
    for i in range(n):
        for j in range(i + 1, n):
            out[iterID, 0] = array[i]
            out[iterID, 1] = array[j]
            iterID += 1
    return out


class ProgressBar:
    """A simple progessbar that can be used with logger."""

    def __init__(self, steps_total, logger, task, interval=1):
        """

        Args:
            steps_total (int): How many steps are there in total
            logger: the "logging.getLogger(...)" object
            interval (int, optional): How often the progress is printed. Defaults to 1.
        """
        self.steps_total = steps_total
        self.time_start = time.time()
        self.time_logged = time.time()
        self.interval = interval
        self.logger = logger
        self.task = task

    def step(self, step, filenames_indices_deleted=None):
        # logs the progress like this:
        # Progress: [93.0647%, 542251/582660] Total-time(h:m:s): 0:0:6 FPS: 77464.42 Deleted-Images: 474
        time_now = time.time()
        if time.time() - self.time_logged > self.interval * 60 or step == 0:
            hours = int((time_now - self.time_start) / (60 * 60)) % (60 * 60)
            minutes = int((time_now - self.time_start) / 60) % 60
            seconds = int((time_now) - self.time_start) % 60
            fps = self.round((step) / (int((time_now - self.time_start)) + 1), 2)
            percentage = self.round((step / self.steps_total) * 100, 4)
            message = f"Progress '{self.task}': [{percentage}%, {step}/{self.steps_total}] Total-time(h:m:s): {hours}:{minutes}:{seconds} FPS: {fps}"
            if filenames_indices_deleted is not None:
                message + f" Deleted-Images: {filenames_indices_deleted.sum()}"
            self.logger.info(message)
            self.time_logged = time.time()

    def round(self, nr, size=2):
        nr = nr * (10**size)
        nr = int(nr)
        nr = nr / (10**size)
        return nr


def show_image(imgs, imgnames, duration=1000):
    """Displayes a list of images and their arrording names

    Args:
        imgs (ordered collections): Images
        imgnames (ordered collections): Image names
        duration (int, optional): How long the image is displayed in ms. Defaults to 1000.
    """
    for img, imgname in zip(imgs, imgnames):
        cv2.imshow(imgname, img)
    cv2.waitKey(duration)
    cv2.destroyAllWindows()


def get_all_images_in_folder(folder, logger=None):
    """Return all images in folder (sorted)

    Args:
        folder (string): Path to image folder

    Returns:
        list: Contains all images in folder
    """

    try:

        if "data/test_images/" in folder:
            if logger is not None:
                logger.warning(f"Default data path is used: {folder}")
            if logger is not None:
                logger.warning(
                    f"Specify own path with config.DATA.PATH_DATASET, or argument '--path-dataset'"
                )

        filenames = []
        for file in os.listdir(os.fsencode(folder)):
            filename = os.fsdecode(file)
            if filename.endswith((".jpeg", ".jpg", ".png", ".gif")):
                filenames.append(filename)
        filenames.sort()
    except TypeError as error:
        if logger is not None:
            logger.error(f"config.DATA.PATH_DATASET must be a string. Got: {folder}")
        raise
    except FileNotFoundError as error:
        if logger is not None:
            logger.error(f"config.DATA.PATH_DATASET is not a valid path. Got: {folder}")
        raise
    return filenames  # now you have the filenames and can do something with them


def delete_file(path, logger):
    try:
        os.remove(path)
    except OSError:
        logger.warning(f"File to be deleted was not found: {path}")
