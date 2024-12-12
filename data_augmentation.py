from keras.preprocessing.image import ImageDataGenerator

def augment_images(images):
    datagen = ImageDataGenerator(
        rotation_range=20,
        width_shift_range=0.2,
        height_shift_range=0.2,
        shear_range=0.2,
        zoom_range=0.2,
        horizontal_flip=True,
        fill_mode='nearest'
    )
    
    augmented_images = []
    
    for img in images:
        img = np.expand_dims(img, axis=0)  # Add batch dimension
        augmented_images.extend(datagen.flow(img, batch_size=1, save_to_dir='augmented_images', save_prefix='aug', save_format='jpg'))
    
    return augmented_images

# Example usage
augmented_images = augment_images(processed_images)
