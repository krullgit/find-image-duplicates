# --------------------------------------------------------
# Duplicate Detection
# Licensed under The GPL2 License [see LICENSE for details]
# Written by Matthes Krull
# --------------------------------------------------------

import logging
import cv2
import os
import time
import sys
import traceback
import numpy as np

from logger import setup_logger
import imaging_interview
import utils


class RemoveDuplicateImages:
    """_summary_
    This class can remove duplicated images.
    """

    def __init__(self):

        # Log all uncaught exceptions
        sys.excepthook = self.log_all_exceptions

        # Set up logger
        self.PATH_APP_BASE = os.path.dirname(os.path.abspath(__file__))
        PATH_LOGGER_CONFIG = os.path.join(
            self.PATH_APP_BASE, "configs", "logger_config.yml"
        )
        PATH_LOGGER_OUT = os.path.join(
            self.PATH_APP_BASE, "logs", "remove_duplicate_images.log"
        )
        setup_logger(PATH_LOGGER_OUT, PATH_LOGGER_CONFIG)
        self.logger = logging.getLogger(__name__)

        # Merge args from commands line and config.py
        _, self.config = utils.parse_args()

        # Get area of image
        self.IMG_AREA = self.config.AUG.IMG_RESCALE_SIZE**2

    def log_all_exceptions(self, type, value, tb):
        """_summary_
            Logs all exceptions.
        """
        for line in traceback.TracebackException(type, value, tb).format(chain=True):
            self.logger.exception(line)
        self.logger.exception(value)

    def remove_duplicates(self):
        """_summary_
        Removes all duplicates in the folder specified in config.DATA.PATH_DATASET
        """

        # Get all names of images in a folder and sort them
        self.logger.info(
            f"Get all image names from the folder path: {self.config.DATA.PATH_DATASET}"
        )
        filenames = utils.get_all_images_in_folder(
            self.config.DATA.PATH_DATASET, self.logger
        )
        self.logger.info(f"Found Images: {len(filenames)}")

        # Get all permutations of the images
        # Creating a list of indices for the filename to speed up look-ups
        filenames_indices = np.arange(0, len(filenames))
        filenames_permutations = utils.pairwise_combs_numba(np.array(filenames_indices))

        # debug
        # path_deleted = os.path.join(config.DATA.PATH_DATASET, "deleted")
        # if not os.path.exists(path_deleted):
        #     os.makedirs(path_deleted)

        # Load all images into memory (crucial for speed)
        self.logger.info(f"Load all images into memory and rescale.")
        imgs_cached = []  # cached images
        imgs_paths = []  # cached images paths
        imgs_sizes = []  # cached images sizes
        for filename in filenames:
            img_path = self.config.DATA.PATH_DATASET + filename
            imgs_paths.append(img_path)
            img = cv2.imread(img_path)

            if img is not None:  # check if the loaded file is valid
                imgs_sizes.append(img.shape[0] * img.shape[1])
                # resize to save memory
                img = cv2.resize(
                    img,
                    (
                        self.config.AUG.IMG_RESCALE_SIZE,
                        self.config.AUG.IMG_RESCALE_SIZE,
                    ),
                    interpolation=cv2.INTER_NEAREST,
                )
            else:  # put placeholder for an invalid file, we will delete them later
                self.logger.info(
                    f"Could not load image. It will be deleted later on: {filename}"
                )
                imgs_sizes.append(0)
            imgs_cached.append(img)
        imgs_cached = np.array(imgs_cached)

        # Set up progress bar
        time_start = time.time()
        progress_bar = utils.ProgressBar(
            len(filenames_permutations), self.logger, minutes=0.1
        )

        assert len(filenames) == len(imgs_cached)
        assert len(filenames) == len(imgs_paths)
        assert len(filenames) == len(imgs_sizes)

        # List, which contains all deleted image indices, so we will not have to process them twice
        filenames_indices_deleted = np.empty((0), dtype=int)

        # Iterate over all permutations
        for i, (img_a_idx, img_b_idx) in enumerate(filenames_permutations):

            progress_bar.step(i, filenames_indices_deleted)

            # If either of the images was deleted already -> skip
            if (filenames_indices_deleted == img_a_idx).any() or (
                filenames_indices_deleted == img_b_idx
            ).any():
                continue

            img_a = imgs_cached[img_a_idx]
            img_b = imgs_cached[img_b_idx]

            # Delete images that are None or too small
            if img_a is None:
                filenames_indices_deleted = np.append(
                    filenames_indices_deleted, img_a_idx
                )
                utils.delete_file(imgs_paths[img_a_idx], self.logger)
                continue
            if img_b is None:
                filenames_indices_deleted = np.append(
                    filenames_indices_deleted, img_b_idx
                )
                utils.delete_file(imgs_paths[img_b_idx], self.logger)
                continue
            if imgs_sizes[img_a_idx] < self.config.DATA.IMG_MIN_AREA:
                filenames_indices_deleted = np.append(
                    filenames_indices_deleted, img_a_idx
                )
                utils.delete_file(imgs_paths[img_a_idx], self.logger)
                continue
            if imgs_sizes[img_b_idx] < self.config.DATA.IMG_MIN_AREA:
                filenames_indices_deleted = np.append(
                    filenames_indices_deleted, img_b_idx
                )
                utils.delete_file(imgs_paths[img_b_idx], self.logger)
                continue

            # utils.show_image([np.append(img_a, img_b, axis=1)], ["original"], duration=2000)

            # preprocess
            img_a = np.array(
                imaging_interview.preprocess_image_change_detection(
                    img_a,
                    self.config.AUG.GAUSSIAN_BLUR_RADIUS_LIST,
                    self.config.AUG.BLACK_MASK,
                )
            )
            img_b = np.array(
                imaging_interview.preprocess_image_change_detection(
                    img_b,
                    self.config.AUG.GAUSSIAN_BLUR_RADIUS_LIST,
                    self.config.AUG.BLACK_MASK,
                )
            )

            # utils.show_image([np.append(img_a, img_b, axis=1)], ["preprocessed"], duration=2000)

            # Calculate the similarity scores
            (
                score,
                res_cnts,
                thresh,
            ) = imaging_interview.compare_frames_change_detection(
                img_a, img_b, self.IMG_AREA * self.config.MODEL.IMG_CONTOUR_THRESHOLD
            )

            # Remove images that are too similar
            img_change_percentage = score / self.IMG_AREA
            if img_change_percentage < self.config.MODEL.IMG_CHANGE_THRESHOLD:
                utils.delete_file(imgs_paths[img_b_idx], self.logger)
                filenames_indices_deleted = np.append(
                    filenames_indices_deleted, img_b_idx
                )

                # debug
                # utils.show_image([np.append(img_a, img_b, axis=1)], [str(img_a_idx)+" "+str(img_b_idx)], duration=200)
                # path = path_deleted +"/"+ str(img_a_idx)+"_"+str(img_b_idx) + ".jpg"
                # img_out = np.append(img_a, img_b, axis=1)
                # img_out = np.append(img_out, contour_img, axis=1)
                # cv2.imwrite(path, img_out)

        # End program after finished
        time_now = time.time()
        hours = int((time_now - time_start) / (60 * 60)) % (60 * 60)
        minutes = int((time_now - time_start) / 60) % 60
        seconds = int((time_now) - time_start) % 60
        self.logger.info(
            f"- Process done - Total-time(h:m:s): {hours}:{minutes}:{seconds} Deleted-Images: {len(filenames_indices_deleted)}/{len(filenames)}"
        )


if __name__ == "__main__":

    duplicate_remover = RemoveDuplicateImages()
    duplicate_remover.remove_duplicates()


# Notes:
# images are videos. so it could be speed up by knowing where one video starts and ends
# sometimes important items like cars are not detected by the contour approach
# task could be distributed with spark or parallelize to use multiple cores
# could use use structural_similarity() of scikit-image
# maybe put option to disable caching to speed up debugging
# I could put everything into a class and call it from a main method to povide better modularity
# could provide a docker file for easy usage
