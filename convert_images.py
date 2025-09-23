from PIL import Image
import os
import numpy as np
import h5py

def construct_volume_from_slices(slices_folder: str, volume_output_path: str, slice_output_path: str, slice_prefix: str, volume_name: str):
    slices_folder = slices_folder.strip('/')

    image_paths = [os.path.join(slices_folder, filename) for filename in sorted(os.listdir(slices_folder))]

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

def create_train_data(img_slice_folder: str, label_slice_folder: str, output_path: str, prefix: str):
    img_slice_folder = img_slice_folder.strip('/')
    label_slice_folder = label_slice_folder.strip('/')

    image_paths = [os.path.join(img_slice_folder, filename) for filename in sorted(os.listdir(img_slice_folder))]
    label_paths = [os.path.join(label_slice_folder, filename) for filename in sorted(os.listdir(label_slice_folder))]
    
    image_data = [np.array(Image.open(path), dtype=np.float32) for path in image_paths]
    label_data = [np.array(Image.open(path), dtype=np.float32) for path in label_paths]


    os.makedirs(output_path, exist_ok=True)
    for index, (i, l) in enumerate(zip(image_data, label_data)):
        train_data_dict = {}
        train_data_dict['image'] = i
        train_data_dict['label'] = l
        
        save_path = os.path.join(output_path, f"{prefix}_slice_{index:04d}.npz")
        print(save_path)
        np.savez_compressed(save_path, **train_data_dict)

if __name__ == '__main__':
    OA_img_folder = './raw_data/OA/imgs'
    OA_label_folder = './raw_data/OA/masks'
    ICA_img_folder = './raw_data/ICA/imgs'
    ICA_label_folder = './raw_data/ICA/masks'
    ICA2_img_folder = './raw_data/ICA2/imgs'
    ICA2_label_folder = './raw_data/ICA2/masks'
    construct_volume_from_slices(slices_folder=OA_img_folder, volume_output_path='./processed_data/OA/volumes/', slice_output_path='./processed_data/OA/slices/image/', slice_prefix='Slice', volume_name='OA_images.h5')
    construct_volume_from_slices(slices_folder=OA_label_folder, volume_output_path='./processed_data/OA/volumes/', slice_output_path='./processed_data/OA/slices/label/', slice_prefix='Slice', volume_name='OA_labels.h5')
    construct_volume_from_slices(slices_folder=ICA_img_folder, volume_output_path='./processed_data/ICA/volumes/', slice_output_path='./processed_data/ICA/slices/image/', slice_prefix='Slice', volume_name='ICA_images.h5')
    construct_volume_from_slices(slices_folder=ICA_label_folder, volume_output_path='./processed_data/ICA/volumes/', slice_output_path='./processed_data/ICA/slices/label/', slice_prefix='Slice', volume_name='ICA_labels.h5')
    construct_volume_from_slices(slices_folder=ICA2_img_folder, volume_output_path='./processed_data/ICA2/volumes/', slice_output_path='./processed_data/ICA2/slices/image/', slice_prefix='Slice', volume_name='ICA2_images.h5')
    construct_volume_from_slices(slices_folder=ICA2_label_folder, volume_output_path='./processed_data/ICA2/volumes/', slice_output_path='./processed_data/ICA2/slices/label/', slice_prefix='Slice', volume_name='ICA2_labels.h5')

    create_train_data(img_slice_folder='./processed_data/OA/slices/image', label_slice_folder='./processed_data/OA/slices/label', output_path='./processed_data/OA/train', prefix='CASE_OA')
    create_train_data(img_slice_folder='./processed_data/ICA/slices/image', label_slice_folder='./processed_data/ICA/slices/label', output_path='./processed_data/ICA/train', prefix='CASE_ICA')
    create_train_data(img_slice_folder='./processed_data/ICA2/slices/image', label_slice_folder='./processed_data/ICA2/slices/label', output_path='./processed_data/ICA2/train', prefix='CASE_ICA2')