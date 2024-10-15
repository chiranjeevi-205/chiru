import os
import random
from datetime import datetime

# Define Task 1: Sampling in 'images' and 'labels' directories
def Detection_sampling(main_dir, sampling_ratio):
    images_dir = os.path.join(main_dir, 'images/')
    labels_dir = os.path.join(main_dir, 'labels/')
    
    # Get all image files from the images directory
    image_files = os.listdir(images_dir)
    
    # Apply random sampling
    num_files_to_sample = int(len(image_files) * sampling_ratio)
    sampled_files = random.sample(image_files, num_files_to_sample)
    
    # Find the set of files to delete (non-sampled files)
    files_to_delete = set(image_files) - set(sampled_files)
    
    # Delete non-sampled images and their corresponding labels
    for image_file in files_to_delete:
        # Define the corresponding label file (assuming same base name with different extension)
        base_name = os.path.splitext(image_file)[0]
        label_file = f"{base_name}.txt"
        
        # Image and label file paths
        image_path = os.path.join(images_dir, image_file)
        label_path = os.path.join(labels_dir, label_file)
    
        # Delete the image file
        if os.path.exists(image_path):
            os.remove(image_path)
            print(f"Deleted image: {image_file}")
    
        # Delete the label file if it exists
        if os.path.exists(label_path):
            os.remove(label_path)
            print(f"Deleted label: {label_file}")
    
    print("Task 1 complete! Non-sampled files have been deleted from 'images' and 'labels' directories.")


# Define Task 2: Sampling in subfolders (e.g., Bird1, Bird2)
def classification_sampling(base_dir, sampling_ratio):
    # Get all subfolders in the base directory (e.g., Bird1, Bird2)
    subfolders = [f for f in os.listdir(base_dir) if os.path.isdir(os.path.join(base_dir, f))]
    
    # Iterate through each subfolder
    for folder in subfolders:
        images_dir = os.path.join(base_dir, folder)
        
        # Get all image files from the current images directory
        image_files = os.listdir(images_dir)
        
        if len(image_files) == 0:
            print(f"No images found in {folder}. Skipping.")
            continue
        
        # Apply random sampling
        num_files_to_sample = int(len(image_files) * sampling_ratio)
        sampled_files = random.sample(image_files, num_files_to_sample)
        
        # Find the set of files to delete (non-sampled files)
        files_to_delete = set(image_files) - set(sampled_files)
        
        # Delete non-sampled images
        for image_file in files_to_delete:
            # Image file path
            image_path = os.path.join(images_dir, image_file)
            
            # Delete the image file
            if os.path.exists(image_path):
                os.remove(image_path)
                print(f"Deleted image: {image_file} from {folder}")
    
    print("Task 2 complete! Non-sampled files have been deleted from all subfolders.")


# # Call both tasks
# today = datetime.now().strftime('%Y%m%d')
# main_dir = f'./{today}'
# base_dir = './output'
# sampling_ratio = 0.2

# Detection_sampling(main_dir, sampling_ratio)
# classification_sampling(base_dir, sampling_ratio)


# import os
# import random
# from datetime import datetime

# # Get today's date
# today = datetime.now().strftime('%Y%m%d')

# # Define the main directory using today's date
# main_dir = f'./{today}'

# # Example of how you can print or use it
# print("Today's date as directory:", main_dir)
