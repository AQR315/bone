from tensorflow.keras.preprocessing.image import img_to_array

def predict_fracture(model, images):
    predictions = []
    
    for img in images:
        img = np.expand_dims(img, axis=0)  # Add batch dimension
        img = img_to_array(img)
        img = img / 255.0  # Normalize the image
        
        pred = model.predict(img)
        predictions.append(pred)
    
    return np.array(predictions)

# Example usage
predictions = predict_fracture(model, processed_images)
