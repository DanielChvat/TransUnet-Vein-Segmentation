import h5py
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider
from matplotlib.colors import ListedColormap

def visualize_h5_volume(h5_path: str, image_key: str = "image", label_key: str = "label"):
    """Visualize image and label slices from a 3D volume with label overlay on image (background transparent)."""
    with h5py.File(h5_path, 'r') as hf:
        if image_key not in hf or label_key not in hf:
            raise KeyError(f"Keys '{image_key}' or '{label_key}' not found in {h5_path}. "
                           f"Available keys: {list(hf.keys())}")
        image_vol = hf[image_key][:]
        label_vol = hf[label_key][:]

    if image_vol.shape[0] != label_vol.shape[0]:
        raise ValueError("Image and label volumes must have the same depth (first dimension).")

    depth = image_vol.shape[0]

    # Build a colormap where 0 = fully transparent, others = jet colors
    jet = plt.cm.jet(np.linspace(0, 1, 256))
    jet[0, -1] = 0  # set alpha=0 for value=0
    cmap_jet_transparent = ListedColormap(jet)

    fig, ax = plt.subplots(figsize=(6, 6))
    plt.subplots_adjust(bottom=0.25)

    slice_idx = 0
    img_display = ax.imshow(image_vol[slice_idx], cmap='gray')
    label_display = ax.imshow(label_vol[slice_idx], cmap=cmap_jet_transparent, alpha=0.7)

    ax.set_title(f"Overlay - Slice {slice_idx}")
    ax.axis("off")

    ax_slider = plt.axes([0.2, 0.1, 0.6, 0.03])
    slider = Slider(ax_slider, 'Slice', 0, depth - 1, valinit=slice_idx, valstep=1)

    def update(val):
        idx = int(slider.val)
        img_display.set_data(image_vol[idx])
        label_display.set_data(label_vol[idx])
        ax.set_title(f"Overlay - Slice {idx}")
        fig.canvas.draw_idle()

    slider.on_changed(update)
    plt.show()


if __name__ == '__main__':
    # Example usage
    visualize_h5_volume('./processed_data/Cube96/volumes/Cube96.h5')
