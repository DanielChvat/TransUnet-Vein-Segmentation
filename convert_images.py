from PIL import Image
import os
import numpy as np
import h5py

def construct_volume_from_slices(slices_folder: str, output_path: str,  volume_name: str):
    slices_folder = slices_folder.strip('/')

    image_paths = [os.path.join(slices_folder, filename) 
                   for filename in sorted(os.listdir(slices_folder))]

    images = [np.array(Image.open(path), dtype=np.float32) for path in image_paths]

    volume_data = np.stack(images, axis=0)

    volume_data = np.clip(volume_data, -125, 275)

    v_min, v_max = volume_data.min(), volume_data.max()
    if v_max != 0: 
        volume_data = (volume_data - v_min) / (v_max - v_min)
    else:
        volume_data = np.zeros_like(volume_data)
    
    os.makedirs(output_path, exist_ok=True)
    with h5py.File(os.path.join(output_path, volume_name), 'w') as hf:
        hf.create_dataset(volume_name, data=volume_data, compression="gzip")
        print(f"Successfully created HDF5 volume at '{output_path}/{volume_name} with dataset {volume_name}")
        


if __name__ == '__main__':
    img_folder = './raw_data/OA/imgs'
    label_folder = './raw_data/OA/masks'
    construct_volume_from_slices(slices_folder=img_folder, output_path='./processed_data/OA/volumes/', volume_name='OA_images.h5')
    construct_volume_from_slices(slices_folder=label_folder, output_path='./processed_data/OA/volumes/', volume_name='OA_labels.h5')