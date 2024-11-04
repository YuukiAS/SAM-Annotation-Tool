import os
import numpy as np
import nibabel as nib
import matplotlib.pyplot as plt
from tqdm import tqdm
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("--prefix", type=str, required=True)


image_dir = "./data/image"
label_dir = "./data/label"

visualize_dir = './img'

if __name__ == "__main__":
    args = parser.parse_args()
    prefix = args.prefix

    # Remove all files in visualize_dir
    for file in os.listdir(visualize_dir):
        file_path = os.path.join(visualize_dir, file)
        os.remove(file_path)

    for file in tqdm(os.listdir(image_dir)):
        if file.startswith(prefix):
            image_path = os.path.join(image_dir, file)
            label_path = os.path.join(label_dir, file)
            file_png_name = file.split('.')[0] + '.png'
            visualize_path = os.path.join(visualize_dir, file_png_name)
            
            image = nib.load(image_path).get_fdata()
            label = nib.load(label_path).get_fdata()

            plt.figure(figsize=(10, 5))
            plt.subplot(1,2,1)
            plt.imshow(image, cmap='gray')
            plt.subplot(1,2,2)
            plt.imshow(image, cmap='gray')
            label_nan = np.where(label == 0, np.nan, label)
            plt.imshow(label_nan, alpha=0.5, cmap='jet')
            plt.savefig(visualize_path)

    print("Done!")