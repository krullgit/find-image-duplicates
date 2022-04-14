from numba import njit
import numpy as np
import time
import cv2
from config import get_config
import argparse
import os



def parse_args():
    """_summary_
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
        required=True,
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
    """_summary_
        https://localcoder.org/fast-combinations-without-replacement-for-arrays-numpy-python
        Computes all permutations between a & a, without repetitions, order not important.
        10x faster than "itertools.combinations"
    Args:
        array (Sequence): The sequence which is permutated

    Returns:
        list: Numpy matrix with shape (n,2) (all permutations)
    """
    # https://localcoder.org/fast-combinations-without-replacement-for-arrays-numpy-python
    # Computes all permutations between a & a, without repetitions.
    # 10x faster than "itertools.combinations"

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
    """_summary_
    A simple progessbar that can be used with logger.
    """

    def __init__(self, steps_total, logger, minutes=1):
        """_summary_

        Args:
            steps_total (int): How many steps are there in total
            logger: the "logging.getLogger(...)" object
            minutes (int, optional): How often the progress is printed. Defaults to 1.
        """
        self.steps_total = steps_total
        self.time_start = time.time()
        self.time_logged = time.time()
        self.minutes = minutes
        self.logger = logger

    def step(self, step, filenames_indices_deleted):
        # logs the progress like this:
        # Progress: [0.0003%, 2/582660] Total-time(h:m:s): 0:0:37 FPS: 6000.00
        time_now = time.time()
        if time.time() - self.time_logged > self.minutes * 60 or step == 0:
            hours = int((time_now - self.time_start) / (60 * 60)) % (60 * 60)
            minutes = int((time_now - self.time_start) / 60) % 60
            seconds = int((time_now) - self.time_start) % 60
            fps = self.round((step) / (int((time_now - self.time_start)) + 1), 2)
            percentage = self.round((step / self.steps_total) * 100, 4)
            self.logger.info(
                f"Progress: [{percentage}%, {step}/{self.steps_total}] Total-time(h:m:s): {hours}:{minutes}:{seconds} FPS: {fps} Deleted-Images: {len(filenames_indices_deleted)}"
            )
            self.time_logged = time.time()

    def round(self, nr, size=2):
        nr = nr * (10**size)
        nr = int(nr)
        nr = nr / (10**size)
        return nr


def show_image(imgs, imgnames, duration=1000):
    """_summary_
        Displayes a list of images and their arrording names

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
    """_summary_
        Return all images in folder (sorted)
    Args:
        folder (string): Path to image folder

    Returns:
        list: Contains all images in folder
    """
    
    try:
        filenames = []
        for file in os.listdir(os.fsencode(folder)):
            filename = os.fsdecode(file)
            if filename.endswith((".jpeg", ".jpg", ".png", ".gif")):
                filenames.append(filename)
        filenames.sort()
    except TypeError as error:
        if logger is not None: logger.error(f"config.DATA.PATH_DATASET must be a string. Got: {folder}")
        raise
    except FileNotFoundError as error:
        if logger is not None: logger.error(f"config.DATA.PATH_DATASET is not a valid path. Got: {folder}")
        raise
    return filenames  # now you have the filenames and can do something with them

def delete_file(path, logger):
    try:
        os.remove(path)
    except OSError:
        logger.warning(f"File to be deleted was not found: {path}")