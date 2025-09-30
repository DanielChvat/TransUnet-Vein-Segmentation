from PIL import Image
import os
import numpy as np
import h5py

TARGET_SIZE = (224, 224)  # (height, width), must be divisible by ViT patch size (e.g. 16)

def resize_to_target_size(img_array: np.ndarray, target_size=TARGET_SIZE, is_label=False):
    """Resizes a 2D array to the target size using interpolation."""
    img = Image.fromarray(img_array)
    if is_label:
        img = img.resize((target_size[1], target_size[0]), Image.NEAREST)   # keep class IDs
    else:
        img = img.resize((target_size[1], target_size[0]), Image.BILINEAR)  # smooth interpolation
    return np.array(img, dtype=img_array.dtype)

def construct_volume_with_labels(img_folder: str, label_folder: str, volume_output_path: str, 
                                 slice_output_path: str, prefix: str, volume_name: str):
    """Builds a volume from image + label slices and saves as HDF5 with keys 'image' and 'label'."""
    img_folder = img_folder.strip('/')
    label_folder = label_folder.strip('/')

    # Sorted paths
    img_paths = [os.path.join(img_folder, f) for f in sorted(os.listdir(img_folder))]
    lbl_paths = [os.path.join(label_folder, f) for f in sorted(os.listdir(label_folder))]

    # Load and resize
    img_slices = [resize_to_target_size(np.array(Image.open(p), dtype=np.float32), is_label=False) for p in img_paths]
    lbl_slices = [resize_to_target_size(np.array(Image.open(p), dtype=np.float32), is_label=True) for p in lbl_paths]

    # Stack to volumes
    img_volume = np.stack(img_slices, axis=0)
    lbl_volume = np.stack(lbl_slices, axis=0)

    # Normalize images (clip HU-like range)
    img_volume = np.clip(img_volume, -125, 275)
    v_min, v_max = img_volume.min(), img_volume.max()
    if v_max != 0:
        img_volume = (img_volume - v_min) / (v_max - v_min)
    else:
        img_volume = np.zeros_like(img_volume)

    # Save slices as TIFF (optional)
    save_preprocessed_slices(img_volume, os.path.join(slice_output_path, "image"), prefix=prefix + "_img")
    save_preprocessed_slices(lbl_volume, os.path.join(slice_output_path, "label"), prefix=prefix + "_lbl")

    # Save as HDF5 with keys
    os.makedirs(volume_output_path, exist_ok=True)
    h5_path = os.path.join(volume_output_path, volume_name)
    with h5py.File(h5_path, 'w') as hf:
        hf.create_dataset('image', data=img_volume, compression="gzip")
        hf.create_dataset('label', data=lbl_volume, compression="gzip")
        print(f"Successfully created HDF5 file at '{h5_path}' with keys ['image', 'label']")

def save_preprocessed_slices(volume_data, output_folder: str, prefix: str = "slice"):
    os.makedirs(output_folder, exist_ok=True)
    for i, slice_data in enumerate(volume_data):
        img = Image.fromarray(slice_data.astype(np.float32))
        filename = os.path.join(output_folder, f"{prefix}_{i:04d}.tiff")
        img.save(filename)
    print(f"Saved {len(volume_data)} slices to '{output_folder}'")

def create_train_data(h5_path: str, output_path: str, prefix: str):
    """Save individual slice-level NPZ files directly from HDF5 volume."""
    with h5py.File(h5_path, 'r') as hf:
        img_volume = hf['image'][:]
        lbl_volume = hf['label'][:]

    assert img_volume.shape[0] == lbl_volume.shape[0], "Mismatch in number of slices"

    os.makedirs(output_path, exist_ok=True)
    for idx, (i, l) in enumerate(zip(img_volume, lbl_volume)):
        train_data_dict = {"image": i, "label": l}
        save_path = os.path.join(output_path, f"{prefix}_slice_{idx:04d}.npz")
        np.savez_compressed(save_path, **train_data_dict)
        print(f"Saved {save_path}")

if __name__ == '__main__':
    datasets = {
        "OA": ("./raw_data/OA/imgs", "./raw_data/OA/masks"),
        "ICA": ("./raw_data/ICA/imgs", "./raw_data/ICA/masks"),
        "ICA2": ("./raw_data/ICA2/imgs", "./raw_data/ICA2/masks"),
        "Cube96": ("./raw_data/Cube96/imgs", "./raw_data/Cube96/masks"),
    }

    for name, (img_folder, lbl_folder) in datasets.items():
        print(f"\nProcessing dataset: {name}")
        volume_path = f'./processed_data/{name}/volumes/'
        h5_name = f'{name}.h5'

        construct_volume_with_labels(
            img_folder=img_folder,
            label_folder=lbl_folder,
            volume_output_path=volume_path,
            slice_output_path=f'./processed_data/{name}/slices/',
            prefix=f'CASE_{name}',
            volume_name=h5_name
        )

        create_train_data(
            h5_path=os.path.join(volume_path, h5_name),
            output_path=f'./processed_data/{name}/train',
            prefix=f'CASE_{name}'
        )
