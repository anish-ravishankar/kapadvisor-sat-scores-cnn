import os
import numpy as np
from PIL import Image
import cv2
import pytesseract


def load_and_preprocess_image(image_path, normalize=True, add_batch_dim=True):
    """
    Load and preprocess a single image.
    
    Args:
        image_path: Path to the image file
        normalize: Whether to normalize pixel values to [0,1]
        add_batch_dim: Whether to add batch dimension for model input
    
    Returns:
        Preprocessed image as numpy array
    """
    img = Image.open(image_path).convert('L')  # Convert to grayscale
    img_array = np.array(img)
    img_array = remove_text_from_image(img_array)
    img = Image.fromarray(img_array.astype(np.uint8))
    img = img.resize((128, 64))  # Resize to match training dimensions
    img_array = np.array(img)
    
    if add_batch_dim:
        img_array = np.expand_dims(img_array, axis=-1)  # Add channel dimension
        img_array = np.expand_dims(img_array, axis=0)   # Add batch dimension
    
    if normalize:
        img_array = img_array / 255.0  # Normalize
        
    return img_array


def load_images_from_folder(folder_path):
    """
    Load and preprocess all images from a folder.
    
    Args:
        folder_path: Path to folder containing images
    
    Returns:
        Tuple of (images array, labels array)
    """
    images = []
    labels = []
    
    for filename in os.listdir(folder_path):
        if filename.endswith('.png'):
            img_path = os.path.join(folder_path, filename)
            # Process image without batch dimension
            img_array = load_and_preprocess_image(
                img_path, 
                normalize=True, 
                add_batch_dim=False
            )
            images.append(img_array)
            
            # Extract label from filename
            label = int(filename.split(' of')[0].split(' ')[-1])
            labels.append(label)
    
    images = np.array(images)
    labels = np.array(labels)
    
    images = np.expand_dims(images, axis=-1)
    return images, labels
    
def remove_text_from_image(image):
    """
    Detects text regions in an image using Tesseract OCR and masks them with white pixels.
    Uses direct text detection instead of bounding boxes for more accurate results.

    Args:
        image: A numpy array representing the grayscale image.
    Returns:
        A numpy array with the text regions masked.
    """
    # Convert to PIL format for OCR processing
    pil_img = Image.fromarray(image)
    data = pytesseract.image_to_data(pil_img, output_type=pytesseract.Output.DICT)
    result = image.copy()

    # Iterate through detected text regions
    n_boxes = len(data['text'])
    for i in range(n_boxes):
        # Filter out empty text and low confidence detections
        if int(data['conf'][i]) > 60:  # Set at 60 based on empirical testing
            (x, y, w, h) = (
                data['left'][i],
                data['top'][i],
                data['width'][i],
                data['height'][i]
            )
            # Create a slightly larger mask region to ensure complete text removal
            padding = 2
            x = max(0, x - padding)
            y = max(0, y - padding)
            w = min(w + 2*padding, image.shape[1] - x)
            h = min(h + 2*padding, image.shape[0] - y)

            # Fill text region with white
            cv2.rectangle(result, (x, y), (x + w, y + h), (255,), -1)

    return result