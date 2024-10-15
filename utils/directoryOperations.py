import glob
import os
import shutil
import cv2

import yaml
from Config import get_env


class directory_operations:
    @staticmethod
    def create_directories(dir_folder):
        """
        Create necessary directories within the specified directory folder.

        Args:
            dir_folder (str): The base directory path.

        Returns:
            None
        """
        frames_path = os.path.join(dir_folder, "frames")
        video_path = os.path.join(dir_folder, "videos")
        image_path = os.path.join(dir_folder, "images")
        croppedimg_path = os.path.join(dir_folder, "croppedimg")

        # Create 'frames' directory if it doesn't exist
        if not os.path.exists(frames_path):
            os.makedirs(frames_path)

        # Create 'videos' directory if it doesn't exist
        if not os.path.exists(video_path):
            os.makedirs(video_path)

        # Create 'images' directory if it doesn't exist
        if not os.path.exists(image_path):
            os.makedirs(image_path)

        # Create 'croppedimg' directory if it doesn't exist
        if not os.path.exists(croppedimg_path):
            os.makedirs(croppedimg_path)

    @staticmethod
    def get_frames_path(sourceId):
        """
        Get the path to the frames directory for a specific source ID.

        Args:
            sourceId (str): The source ID.

        Returns:
            str: The path to the frames directory.
        """
        config = get_env.Settings()
        return f"./{config.rootPath}/{sourceId}/frames/"

    @staticmethod
    def get_cropped_images_path(sourceId):
        """
        Get the path to the frames directory for a specific source ID.

        Args:
            sourceId (str): The source ID.

        Returns:
            str: The path to the frames directory.
        """
        config = get_env.Settings()
        return f"./{config.rootPath}/{sourceId}/croppedimg/"

    @staticmethod
    def get_images_path(sourceId):
        """
        Get the path to the images directory for a specific source ID.

        Args:
            sourceId (str): The source ID.

        Returns:
            str: The path to the images directory.
        """
        config = get_env.Settings()
        return f"./{config.rootPath}/{sourceId}/images/"

    @staticmethod
    def get_videos_path(sourceId):
        """
        Get the path to the videos directory for a specific source ID.

        Args:
            sourceId (str): The source ID.

        Returns:
            str: The path to the videos directory.
        """
        config = get_env.Settings()
        return f"./{config.rootPath}/{sourceId}/videos/"

    @staticmethod
    def get_all_images(full_path):
        """
        Get a list of all image files in the specified directory.

        Args:
            full_path (str): The full path to the directory.

        Returns:
            list: List of image file paths.
        """
        image_files = (
            glob.glob(os.path.join(full_path, "*.jpg"))
            + glob.glob(os.path.join(full_path, "*.jpeg"))
            + glob.glob(os.path.join(full_path, "*.png"))
        )
        return image_files

    @staticmethod
    def get_config_map(configfile):
        """
        Read and return the content of a YAML configuration file.

        Args:
            configfile (str): The name of the configuration file (without the extension).

        Returns:
            dict: Dictionary containing the configuration information.
        """
        with open(f"./config/{configfile}.yaml", "r") as file:
            modelInfo = yaml.safe_load(file)
        return modelInfo

    @staticmethod
    def remove_from_directory(full_path):
        """
        Remove a directory and its contents.

        Args:
            full_path (str): The full path to the directory.

        Returns:
            None
        """
        shutil.rmtree(full_path)

    @staticmethod
    def crop_and_save_image(image_path, bbox, save_path):
        img = cv2.imread(image_path)
        xmin, ymin, xmax, ymax = bbox
        cropped_img = img[int(ymin) : int(ymax), int(xmin) : int(xmax)]
        output_dir = os.path.dirname(save_path)
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)

        # Save the cropped image
        cv2.imwrite(save_path, cropped_img)
