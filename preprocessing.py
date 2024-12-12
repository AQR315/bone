def preprocess_images(images, target_size=(128, 128)):
    processed_images = []
    
    for img in images:
        # Binarization
        _, binarized_img = cv2.threshold(img, 128, 255, cv2.THRESH_BINARY)
        
        # Noise removal (using GaussianBlur)
        denoised_img = cv2.GaussianBlur(binarized_img, (5, 5), 0)
        
        # Resize image
        resized_img = cv2.resize(denoised_img, target_size)
        
        processed_images.append(resized_img)
    
    return np.array(processed_images)

# Example usage
processed_images = preprocess_images(images)
