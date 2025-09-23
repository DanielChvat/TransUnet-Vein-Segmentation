import h5py
import numpy as np
import matplotlib.pyplot as plt

import h5py
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider

def visualize_h5_volume(h5_path: str):
    dataset_name = h5_path.split('/')[-1]
    with h5py.File(h5_path, 'r') as hf:
        volume = hf[dataset_name][:]
    
    depth = volume.shape[0]

    fig, ax = plt.subplots()
    plt.subplots_adjust(bottom=0.25)

    slice_idx = 0
    img = ax.imshow(volume[slice_idx], cmap='gray')
    ax.set_title(f"Slice {slice_idx}")
    ax.axis("off")

    ax_slider = plt.axes([0.2, 0.1, 0.6, 0.03])
    slider = Slider(ax_slider, 'Slice', 0, depth - 1, valinit=slice_idx, valstep=1)

    def update(val):
        idx = int(slider.val)
        img.set_data(volume[idx])
        ax.set_title(f"Slice {idx}")
        fig.canvas.draw_idle()

    slider.on_changed(update)
    plt.show()


if __name__ == '__main__':
    visualize_h5_volume('./processed_data/ICA2/volumes/ICA2_images.h5')