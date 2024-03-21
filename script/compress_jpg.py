import os
from PIL import Image

def batch_resize_images(input_folder, output_folder, scale_factor=0.5):
    """
    Batch resizes images in a folder to a fraction of their original size.

    :param input_folder: Folder containing images to resize.
    :param output_folder: Folder to save resized images.
    :param scale_factor: Factor by which to scale the images (0 < scale_factor <= 1).
    """
    # Create the output folder if it doesn't exist
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    # Iterate through all files in the input folder
    for filename in os.listdir(input_folder):
        if filename.lower().endswith((".jpg", ".jpeg", ".png", ".bmp", ".gif", ".tiff")):
            file_path = os.path.join(input_folder, filename)
            output_path = os.path.join(output_folder, filename)
            # Open the image and resize it
            with Image.open(file_path) as img:
                # Calculate the new size
                new_size = tuple(int(dim * scale_factor) for dim in img.size)
                resized_img = img.resize(new_size, Image.LANCZOS)
                # Save the resized image
                resized_img.save(output_path)

def batch_compress_jpg(input_folder, output_folder, quality=85):
    """
    Batch compresses JPEG images in a folder.

    :param input_folder: Folder containing JPEG images to compress.
    :param output_folder: Folder to save compressed images.
    :param quality: Compression quality (1-100, where 100 is least compression).
    """
    # Create the output folder if it doesn't exist
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

        # Iterate through all files in the input folder
    for filename in os.listdir(input_folder):
        if filename.lower().endswith(".jpg") or filename.lower().endswith(".jpeg"):
            file_path = os.path.join(input_folder, filename)
            output_path = os.path.join(output_folder, filename)

            # Open the image and compress it
            with Image.open(file_path) as img:
                img.save(output_path, "JPEG", quality=quality)

if __name__ == '__main__':

    # Example usage
    batch_compress_jpg("../data/debug", "../data/result_compress", quality=20)
    # Example usage
    batch_resize_images("../data/result_compress", "../data/result_compress_", scale_factor=0.25)
