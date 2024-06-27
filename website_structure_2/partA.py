import requests
from PIL import Image
import torch
from transformers import DPTFeatureExtractor, DPTForSemanticSegmentation
import numpy as np
import tensorflow as tf
from io import BytesIO

# Function to preprocess the image
def process_image(image_path):
    img = Image.open(image_path).convert("RGB")
    return img

# Function to calculate class percentages
def calculate_class_percentages(predicted_labels, classes):
    total_pixels = predicted_labels.size
    percentages = {class_id: np.sum(predicted_labels == class_id) / total_pixels * 100 for class_id in classes.keys()}
    return percentages

# Function to preprocess patches from the building segment for the material classifier
def extract_patches(building_image, patch_size=128, stride=64):
    building_array = np.array(building_image)
    h, w, _ = building_array.shape
    patches = []
    for i in range(0, h - patch_size + 1, stride):
        for j in range(0, w - patch_size + 1, stride):
            patch = building_array[i:i+patch_size, j+j+patch_size]
            if np.sum(patch) > 0:  # Avoid patches with only background
                patches.append(patch)
    patches = np.array(patches) / 255.0
    return patches

# Load segmentation model and feature extractor
feature_extractor = DPTFeatureExtractor.from_pretrained("Intel/dpt-large-ade")
segmentation_model = DPTForSemanticSegmentation.from_pretrained("Intel/dpt-large-ade")

# Load your material texture classifier model
classifier_model_url = "https://github.com/mohammadabid-macad/AIAStudioG03/raw/65f48a58e1f1ea1c8ac387facfa78a6ba20b467d/models/material_texture_classifier.keras"
response = requests.get(classifier_model_url)
classifier_model_path = 'material_texture_classifier.keras'
with open(classifier_model_path, 'wb') as file:
    file.write(response.content)

# Define a custom function to handle input shapes and load the model
def custom_load_model(model_path):
    model = tf.keras.models.load_model(model_path, compile=False)
    input_shape = (None, 128, 128, 3)
    model.build(input_shape)
    return model

classifier_model = custom_load_model(classifier_model_path)

def process_uploaded_image(image_path):
    # Preprocess the image
    original_image = process_image(image_path)
    inputs = feature_extractor(images=original_image, return_tensors="pt")

    # Perform segmentation
    with torch.no_grad():
        outputs = segmentation_model(**inputs)
        logits = outputs.logits
        upsampled_logits = torch.nn.functional.interpolate(logits, size=original_image.size[::-1], mode="bilinear", align_corners=False)
        predicted_labels = upsampled_logits.argmax(dim=1).squeeze().numpy()

    # Define the class mappings
    foliage_classes = [8, 10, 12, 17, 18, 19, 20]  # Example foliage classes, adjust as necessary
    building_class = 1
    sky_class = 22

    # Create a new map for the desired classes
    new_predicted_labels = np.zeros_like(predicted_labels)
    new_predicted_labels[np.isin(predicted_labels, foliage_classes)] = 1  # Foliage
    new_predicted_labels[predicted_labels == building_class] = 2  # Building
    new_predicted_labels[predicted_labels == sky_class] = 3  # Sky

    # Define the colors and names for each class
    classes = {
        0: {"name": "Background", "color": [0, 0, 0]},
        1: {"name": "Foliage", "color": [107, 142, 35]},
        2: {"name": "Building", "color": [128, 64, 128]},
        3: {"name": "Sky", "color": [135, 206, 235]}
    }

    # Calculate percentage for each class
    percentages = calculate_class_percentages(new_predicted_labels, classes)

    # Print the percentages with class names
    for class_id, percentage in percentages.items():
        print(f"{classes[class_id]['name']}: {percentage:.2f}%")

    # Visualize the segmentation
    colored_segmentation = np.zeros((new_predicted_labels.shape[0], new_predicted_labels.shape[1], 3), dtype=np.uint8)
    for class_id, class_info in classes.items():
        colored_segmentation[new_predicted_labels == class_id] = class_info["color"]

    plt.figure(figsize=(10, 10))
    plt.imshow(original_image)
    plt.imshow(colored_segmentation, alpha=0.6)
    plt.axis('off')
    plt.show()

    # Extract the building segment
    building_mask = (new_predicted_labels == 2)
    building_segment = np.array(original_image) * np.repeat(building_mask[:, :, np.newaxis], 3, axis=2)

    # Convert the building segment back to an image
    building_image = Image.fromarray(building_segment.astype('uint8'))

    # Visualize the building segment
    plt.figure(figsize=(10, 10))
    plt.imshow(building_image)
    plt.axis('off')
    plt.show()

    # Extract patches
    patches = extract_patches(building_image)

    # Debugging: Print the shape of the patches
    print(f"Patches shape: {patches.shape}")

    # Classify each patch
    material_predictions = classifier_model.predict(patches)

    # Debugging: Print the shape of the predictions
    print(f"Material predictions shape: {material_predictions.shape}")

    material_classes = ['brick', 'ceramic', 'glass', 'metal', 'paint', 'stone', 'tile', 'wood']
    material_counts = {material: 0 for material in material_classes}

    for pred in material_predictions:
        argmax_index = np.argmax(pred)
        # Debugging: Print the argmax index and the corresponding prediction
        print(f"Argmax index: {argmax_index}, Prediction: {pred}")
        if argmax_index < len(material_classes):
            material_class = material_classes[argmax_index]
            material_counts[material_class] += 1

    # Calculate percentages for each material in the building segment
    total_building_patches = len(patches)
    material_percentages = {material: (count / total_building_patches) * 100 for material, count in material_counts.items()}

    # Print the material classification results with percentages
    for material, percentage in material_percentages.items():
        print(f"{material}: {percentage:.2f}%")

    return material_percentages, "/static/Assets/01_segmentation.png", "/static/Assets/02_Classifier.png"

