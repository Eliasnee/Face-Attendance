import os
import cv2
import albumentations as A
from tqdm import tqdm

# === Configuration ===
input_root = r"known_faces"
output_root = r"known_faces_augmented"
num_augments = 30  # Number of augmentations per image

# === Albumentations Augmentation Pipeline ===
augmentations = A.Compose([
    A.Rotate(limit=45, p=0.8),
    A.Perspective(scale=(0.05, 0.1), p=0.6),
    A.Downscale(scale_min=0.3, scale_max=0.7, p=0.5),
    A.MotionBlur(blur_limit=7, p=0.5),
    A.GaussNoise(var_limit=(10.0, 50.0), p=0.5),
    A.RandomBrightnessContrast(brightness_limit=0.4, contrast_limit=0.4, p=0.5),
    A.HorizontalFlip(p=0.5),
    A.VerticalFlip(p=0.3),
    A.CoarseDropout(max_height=32, max_width=32,
                        min_holes=1, max_holes=3, fill_value=0, p=0.5),
    A.ImageCompression(quality_lower=30, quality_upper=70, p=0.4),
], p=1.0)

# === Process Each Person Folder ===
for person_name in tqdm(os.listdir(input_root), desc="Processing People"):
    person_input_path = os.path.join(input_root, person_name)
    person_output_path = os.path.join(output_root, person_name)
    
    if not os.path.isdir(person_input_path):
        continue  # Skip files

    os.makedirs(person_output_path, exist_ok=True)

    # Process each image in the person's folder
    for filename in os.listdir(person_input_path):
        if filename.lower().endswith(('.jpg', '.jpeg', '.png')):
            img_path = os.path.join(person_input_path, filename)
            image = cv2.imread(img_path)
            if image is None:
                continue

            for i in range(num_augments):
                augmented = augmentations(image=image)['image']
                new_filename = f"{os.path.splitext(filename)[0]}_aug{i+1}.jpg"
                cv2.imwrite(os.path.join(person_output_path, new_filename), augmented)
