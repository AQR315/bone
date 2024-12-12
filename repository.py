import os
import cv2
import numpy as np

def load_images_from_directory(directory):
    images = []
    labels = []
    
    for label in os.listdir(directory):
        label_path = os.path.join(directory, label)
        
        if os.path.isdir(label_path):
            for img_name in os.listdir(label_path):
                img_path = os.path.join(label_path, img_name)
                img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
                images.append(img)
                labels.append(label)
    
    return np.array(images), np.array(labels)

# Example usage
images, labels = load_images_from_directory('/path/to/repository')
