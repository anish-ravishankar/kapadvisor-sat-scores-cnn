import tensorflow as tf
from utils import load_and_preprocess_image  
import warnings
warnings.filterwarnings("ignore")

def predict(model, image_path: str):
    """Run model prediction on extracted section.
    
    Args:
        model: The loaded Keras model
        image_path: Path to the image file
    
    Returns:
        The predicted label
    """
    
    img_array = load_and_preprocess_image(image_path)
    prediction = model.predict(img_array)
    predicted_label = round(prediction[0][0])
    return predicted_label

def validate_samples(model_path: str, image_path: str):
    """
    Validate samples using loaded model.
    
    Args:
        model_path: Path to saved model
        image_path: Path to the image file
    
    Returns:
        The predicted label or 0 if there was an error loading the model
    """
    try:
        model = tf.keras.models.load_model(model_path)
        prediction = predict(model, image_path)
        return prediction
    except Exception as e:
        print(f"Error loading model: {e}")
        return 0
    
if __name__ == "__main__":
    
    model_path = "model_v1_20250109.h5"
    image_path = "CroppedImages/*.png"
    import glob
    for file in glob.glob(image_path):
        prediction = validate_samples(model_path, file)
        print(f"Filename: {file} Label: {prediction}")