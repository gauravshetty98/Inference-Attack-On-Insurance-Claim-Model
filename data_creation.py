import os
import pandas as pd
import shutil

# Define paths
src_folder = "src/image/"        # Source folder containing images
data_file = "src/data 2.csv"      # Path to the CSV file
dest_folder = "renamed_images/" # Destination folder for renamed images

# Create destination folder if it doesn't exist
os.makedirs(dest_folder, exist_ok=True)

# Load the CSV file
data = pd.read_csv(data_file)

# Assuming your CSV has columns 'image_name' and 'class_name'
for index, row in data.iterrows():
    image_name = row['image']   # e.g., "1.jpeg"
    class_name = row['classes']   # e.g., "car_damage"
    
    # Construct source and destination file paths
    src_image_path = os.path.join(src_folder, image_name)
    new_image_name = f"{class_name}_{image_name}"  # e.g., "car_damage_1.jpeg"
    dest_image_path = os.path.join(dest_folder, new_image_name)
    
    # Copy and rename the image
    if os.path.exists(src_image_path):
        shutil.copy(src_image_path, dest_image_path)
        print(f"Renamed and moved: {src_image_path} -> {dest_image_path}")
    else:
        print(f"Image not found: {src_image_path}")

print("Image renaming complete!")
