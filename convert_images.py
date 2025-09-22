from PIL import Image
import os
import numpy as np
import h5py

def construct_volume_from_slices(slices_folder: str, volume_output_path: str, slice_output_path: str, slice_prefix: str, volume_name: str):
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
    
    save_preprocessed_slices(volume_data=volume_data, output_folder=slice_output_path, prefix=slice_prefix)

    os.makedirs(volume_output_path, exist_ok=True)
    with h5py.File(os.path.join(volume_output_path, volume_name), 'w') as hf:
        hf.create_dataset(volume_name, data=volume_data, compression="gzip")
        print(f"Successfully created HDF5 volume at '{volume_output_path}/{volume_name} with dataset {volume_name}")

def save_preprocessed_slices(volume_data, output_folder: str, prefix: str = "slice"):
    os.makedirs(output_folder, exist_ok=True)
    for i, slice_data in enumerate(volume_data):
        img = Image.fromarray(slice_data.astype(np.float32))
        filename = os.path.join(output_folder, f"{prefix}_{i:04d}.tiff")
        img.save(filename)
    print(f"Saved {len(volume_data)} slices to '{output_folder}'")




if __name__ == '__main__':
    img_folder = './raw_data/OA/imgs'
    label_folder = './raw_data/OA/masks'
    construct_volume_from_slices(slices_folder=img_folder, volume_output_path='./processed_data/OA/volumes/', slice_output_path='./processed_data/OA/slices/image/', slice_prefix='Slice', volume_name='OA_images.h5')
    construct_volume_from_slices(slices_folder=label_folder, volume_output_path='./processed_data/OA/volumes/', slice_output_path='./processed_data/OA/slices/label/', slice_prefix='Slice', volume_name='OA_labels.h5')