import os
import cv2
import numpy as np

image_folder_norm = '/home/zty/project/disease/handNorm'
image_folder_dise = '/home/zty/project/disease/handDise'

output_folder = '/home/zty/project/disease/dataset'

# Define the size of images
IMG_WIDTH = 224
IMG_HEIGHT = 224

# Define the number of images in each set
N = 50

# Define the output file names
train_file = 'train.npy'
test_file = 'test.npy'

# Collect the list of image paths
image_paths_norm = [os.path.join(image_folder_norm, f) for f in os.listdir(image_folder_norm) if f.endswith('.jpg')]
image_paths_dise = [os.path.join(image_folder_dise, f) for f in os.listdir(image_folder_dise) if f.endswith('.jpg')]

# Shuffle the image paths
np.random.shuffle(image_paths_norm)
np.random.shuffle(image_paths_dise)

# Select the first N images from each set
image_paths_norm = image_paths_norm[:N]
image_paths_dise = image_paths_dise[:N]

# Combine the image paths
image_paths = image_paths_norm + image_paths_dise

# Define the labels (0 for normal and 1 for diseased)
labels = np.zeros((2*N,), dtype=np.int32)
labels[N:] = 1

# Shuffle the images and labels
idxs = np.arange(2*N)
np.random.shuffle(idxs)
image_paths = [image_paths[i] for i in idxs]
labels = labels[idxs]

# Create the output folder if it does not exist
if not os.path.exists(output_folder):
    os.makedirs(output_folder)

# Initialize the data array
data = np.zeros((2*N, IMG_HEIGHT, IMG_WIDTH, 3), dtype=np.uint8)

# Load the images and resize them
for i, path in enumerate(image_paths):
    img = cv2.imread(path)

    img = cv2.resize(img, (IMG_WIDTH, IMG_HEIGHT))
    cv2.imshow('MediaPipe Hands', img)
    if cv2.waitKey(100) & 0xFF == 27:
        break
    data[i] = img

# Save the data to the output files
np.save(os.path.join(output_folder, train_file), {'images': data[:75], 'labels': labels[:75]})
np.save(os.path.join(output_folder, test_file), {'images': data[75:], 'labels': labels[75:]})
