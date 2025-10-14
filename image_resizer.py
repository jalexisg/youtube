#!/usr/bin/env python3
from PIL import Image
import os

def resize_to_16_9(image_path, output_path):
    """
    Resize an image to 16:9 aspect ratio while maintaining its quality
    """
    try:
        # Open the image
        with Image.open(image_path) as img:
            # Get the original size
            width, height = img.size
            
            # Calculate target dimensions (16:9 aspect ratio)
            target_ratio = 16/9
            current_ratio = width/height
            
            if current_ratio > target_ratio:
                # Image is too wide, adjust width
                new_width = int(height * target_ratio)
                new_height = height
                # Calculate crop box
                left = (width - new_width) // 2
                top = 0
                right = left + new_width
                bottom = height
            else:
                # Image is too tall, adjust height
                new_width = width
                new_height = int(width / target_ratio)
                # Calculate crop box
                left = 0
                top = (height - new_height) // 2
                right = width
                bottom = top + new_height
            
            # Crop the image
            img_cropped = img.crop((left, top, right, bottom))
            
            # Save the cropped image
            img_cropped.save(output_path, quality=95, optimize=True)
            print(f"Successfully processed: {os.path.basename(image_path)}")
            
    except Exception as e:
        print(f"Error processing {image_path}: {str(e)}")

def process_images_in_folder(input_folder="images", output_folder="images_16_9"):
    """
    Process all images in the input folder and save them to the output folder
    """
    # Create output folder if it doesn't exist
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)
        print(f"Created output folder: {output_folder}")
    
    # Supported image formats
    supported_formats = ('.png', '.jpg', '.jpeg', '.webp', '.bmp')
    
    # Process each image in the input folder
    if not os.path.exists(input_folder):
        print(f"Error: Input folder '{input_folder}' does not exist.")
        print("Creating the 'images' folder for you...")
        os.makedirs(input_folder)
        print("Please place your images in the 'images' folder and run the script again.")
        return
    
    found_images = False
    for filename in os.listdir(input_folder):
        if filename.lower().endswith(supported_formats):
            found_images = True
            input_path = os.path.join(input_folder, filename)
            output_path = os.path.join(output_folder, f"16_9_{filename}")
            resize_to_16_9(input_path, output_path)
    
    if not found_images:
        print("No images found in the input folder.")
        print("Supported formats:", ", ".join(supported_formats))

if __name__ == "__main__":
    print("Image Resizer - 16:9 Aspect Ratio Converter")
    print("===========================================")
    process_images_in_folder()
    print("\nDone! Check the 'images_16_9' folder for the converted images.")
