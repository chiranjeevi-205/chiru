import os
import requests
from label_studio_sdk import Client
import base64
from datetime import datetime

# Function to load class names from a classes.txt file
def load_class_names(classes_file_path):
    #print(classes_file_path)
    #classes_file_path="/home/eizen/Desktop/Edge_devices/Lset-Attention-Inference/20241001/classes.txt"
    with open(classes_file_path, 'r') as f:
        return f.read().splitlines()

# Function to create a dynamic label config based on class names
def create_label_config_from_file(class_names):
    label_config = '''
    <View>
      <Image name="image" value="$image"/>
      <RectangleLabels name="bbox" toName="image">
    '''
    for class_name in class_names:
        label_config += f'    <Label value="{class_name.strip()}" background="#ff0000"/>\n'
    label_config += '''  </RectangleLabels>
    </View>
    '''
    return label_config

# Function to create a dynamic label config based on folder names (auto-detect classes)
def create_label_config_from_folders(class_names):
    label_config = '<View>\n<Image name="image" value="$image"/>\n<Choices name="label" toName="image">\n'
    for class_name in class_names:
        label_config += f'    <Choice value="{class_name.strip()}"/>\n'
    label_config += '</Choices>\n</View>'
    return label_config

# Function to create a new project in Label Studio
def create_project(label_studio_url, api_key, project_title, label_config, description):
    headers = {
        'Authorization': f'Token {api_key}',
        'Content-Type': 'application/json',
    }

    data = {
        'title': project_title,
        'description': description,
        'label_config': label_config
    }

    response = requests.post(f'{label_studio_url}/api/projects', headers=headers, json=data)

    if response.status_code == 201:
        project_id = response.json()['id']
        print(f'Project created successfully with ID: {project_id}')
        return project_id
    else:
        print(f'Failed to create project: {response.status_code}, {response.text}')
        return None

# Function to initialize Label Studio Client
def initialize_client(label_studio_url, api_key):
    client = Client(url=label_studio_url, api_key=api_key)
    client.session = requests.Session()
    client.session.headers.update({'Authorization': f'Token {api_key}'})
    client.session.timeout = 200  # Set session timeout to 120 seconds
    return client

# Function to read image and annotation files, and convert annotations to Label Studio format
def prepare_tasks(images_folder, labels_folder, class_names):
    tasks = []
    for image_name in os.listdir(images_folder):
        image_path = os.path.join(images_folder, image_name)
        label_path = os.path.join(labels_folder, image_name.replace('.jpg', '.txt'))  # Assuming .txt annotation

        if os.path.isfile(image_path) and os.path.isfile(label_path):
            with open(image_path, 'rb') as image_file:
                image_data = base64.b64encode(image_file.read()).decode('utf-8')

            with open(label_path, 'r') as label_file:
                label_data = label_file.readlines()

            annotations = []
            for line in label_data:
                line = line.strip()

                if not line:
                    continue

                values = line.split()
                if len(values) != 5:
                    print(f"Skipping invalid line in {label_path}: {line}")
                    continue

                class_id, x_center, y_center, width, height = map(float, values)

                annotation = {
                    'from_name': 'bbox',
                    'to_name': 'image',
                    'type': 'rectanglelabels',
                    'value': {
                        'x': (x_center - width / 2) * 100,
                        'y': (y_center - height / 2) * 100,
                        'width': width * 100,
                        'height': height * 100,
                        'rectanglelabels': [class_names[int(class_id)]]
                    },
                    'origin': 'manual'
                }
                annotations.append(annotation)

            task = {
                'data': {
                    'image': f'data:image/jpeg;base64,{image_data}'
                },
                'annotations': [
                    {
                        'result': annotations
                    }
                ]
            }

            tasks.append(task)
    return tasks

# Function to process images and create tasks based on folder structure
def create_tasks_from_folders(base_path, class_names):
    tasks = []
    for folder_name in class_names:
        folder_path = os.path.join(base_path, folder_name)
        for image_name in os.listdir(folder_path):
            image_path = os.path.join(folder_path, image_name)

            if os.path.isfile(image_path):
                _, ext = os.path.splitext(image_name)
                if ext.lower() in {'.jpg', '.jpeg', '.png', '.gif'}:
                    try:
                        with open(image_path, 'rb') as image_file:
                            image_data = base64.b64encode(image_file.read()).decode('utf-8')

                        # Prepare task with image data and annotation
                        task = {
                            'data': {
                                'image': f'data:image/jpeg;base64,{image_data}'
                            },
                            'annotations': [
                                {
                                    'result': [
                                        {
                                            'from_name': 'label',
                                            'to_name': 'image',
                                            'type': 'choices',
                                            'value': {
                                                'choices': [folder_name]
                                            }
                                        }
                                    ]
                                }
                            ]
                        }
                        tasks.append(task)

                    except Exception as e:
                        print(f"Failed to process {image_path}: {e}")
    return tasks

# Function to upload tasks to Label Studio in batches
def upload_tasks_in_batches(project, tasks, batch_size=5):
    if tasks:
        for i in range(0, len(tasks), batch_size):
            batch = tasks[i:i + batch_size]
            try:
                project.import_tasks(batch)
                print(f'Successfully uploaded batch {i//batch_size + 1}')
            except Exception as e:
                print(f'Failed to upload batch {i//batch_size + 1}: {e}')
    else:
        print('No tasks to upload.')

# Main function to handle the full process (from file-based class names)
def Detection_Anntation(label_studio_url, api_key, base_folder_path, project_title):
    images_folder = os.path.join(base_folder_path, 'images')
    labels_folder = os.path.join(base_folder_path, 'labels')
    classes_file_path = os.path.join(base_folder_path, 'classes.txt')
    print(images_folder)
    print(labels_folder)
    print(classes_file_path)

    # Load class names
    class_names = load_class_names(classes_file_path)

    # Create dynamic label config
    label_config = create_label_config_from_file(class_names)

    # Create a new project in Label Studio
    project_id = create_project(label_studio_url, api_key, project_title, label_config, "This is a new object detection project.")

    if project_id is None:
        return  # Exit if project creation failed

    # Initialize Label Studio Client
    client = initialize_client(label_studio_url, api_key)

    # Fetch the project
    project = client.get_project(project_id)

    # Prepare tasks (images and annotations)
    tasks = prepare_tasks(images_folder, labels_folder, class_names)

    # Upload tasks to Label Studio in batches
    upload_tasks_in_batches(project, tasks)

# Main function to handle the full process (from folder-based class names)
def classification_annnotation(label_studio_url, api_key, project_title, base_path):
    # Step 1: Detect class names from folder structure
    class_names = [folder_name for folder_name in os.listdir(base_path) if os.path.isdir(os.path.join(base_path, folder_name))]

    if not class_names:
        print(f"No class names detected in {base_path}. Exiting.")
        return

    # Step 2: Create a dynamic label config based on the class names
    label_config = create_label_config_from_folders(class_names)

    # Step 3: Create a new project in Label Studio
    project_id = create_project(label_studio_url, api_key, project_title, label_config, "This is a new project with automatically detected class names.")

    if project_id is None:
        return  # Exit if project creation failed

    # Step 4: Initialize Label Studio Client
    client = initialize_client(label_studio_url, api_key)
    project = client.get_project(project_id)

    # Step 5: Create tasks by reading the images and annotations
    tasks = create_tasks_from_folders(base_path, class_names)

    # Step 6: Upload tasks to Label Studio in batches
    if tasks:
        upload_tasks_in_batches(project, tasks, batch_size=5)  # Batch size can be adjusted as needed
    else:
        print('No tasks to upload.')

# Call the main function with appropriate parameters (choose one of the following)
if __name__ == "__main__":
    LABEL_STUDIO_URL = 'https://lset-labelstudio.eizen.ai'
    API_KEY = '78ee35f891b2c71089ef1e07b60ade5cdfcf5bee'
    
    # For file-based class names
    today = datetime.now().strftime('%Y%m%d')
    BASE_FOLDER_PATH = f'./{today}'
    PROJECT_TITLE = "Detection"
    Detection_Anntation(LABEL_STUDIO_URL, API_KEY, BASE_FOLDER_PATH, PROJECT_TITLE)

    # For folder-based class names
    PROJECT_TITLE = "classification_data"
    BASE_PATH = "./output"  # Path to the folder with class-based subfolders
    classification_annnotation(LABEL_STUDIO_URL, API_KEY, PROJECT_TITLE, BASE_PATH)
