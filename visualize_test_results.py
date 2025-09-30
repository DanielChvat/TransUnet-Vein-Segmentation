import nibabel as nib
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider

def load_nii(path):
    return nib.load(path).get_fdata()

def visualize(img_path, gt_path, pred_path):
    # Load volumes
    img = load_nii(img_path)   # (H, W, D)
    gt = load_nii(gt_path)
    pred = load_nii(pred_path)

    img = np.rot90(img, k=1, axes=(0, 1))
    gt = np.rot90(gt, k=1, axes=(0, 1))
    pred = np.rot90(pred, k=1, axes=(0, 1))

    num_slices = img.shape[-1]

    # Setup figure
    fig, axes = plt.subplots(1, 2, figsize=(10, 5))
    plt.subplots_adjust(bottom=0.25)

    slice_idx = 0
    img_disp1 = axes[0].imshow(np.flipud(img[..., slice_idx]), cmap="gray")
    gt_disp = axes[0].imshow(np.flipud(gt[..., slice_idx]), cmap="gray", alpha=0.4)
    axes[0].set_title("Ground Truth")

    img_disp2 = axes[1].imshow(np.flipud(img[..., slice_idx]), cmap="gray")
    pred_disp = axes[1].imshow(np.flipud(pred[..., slice_idx]), cmap="gray", alpha=0.4)
    axes[1].set_title("Prediction")

    # Slider
    ax_slider = plt.axes([0.25, 0.1, 0.5, 0.03])
    slider = Slider(ax_slider, "Slice", 0, num_slices - 1,
                    valinit=slice_idx, valstep=1)

    def update(val):
        idx = int(slider.val)
        img_disp1.set_data(np.flipud(img[..., idx]))
        gt_disp.set_data(np.flipud(gt[..., idx]))
        img_disp2.set_data(np.flipud(img[..., idx]))
        pred_disp.set_data(np.flipud(pred[..., idx]))
        fig.canvas.draw_idle()


    slider.on_changed(update)
    plt.show()


if __name__ == "__main__":
    img_path = "predictions/TU_Veinseg224/TU_pretrain_R50-ViT-B_16_skip3_bs4_224/ICA2_img.nii.gz"
    gt_path = "predictions/TU_Veinseg224/TU_pretrain_R50-ViT-B_16_skip3_bs4_224/ICA2_gt.nii.gz"
    pred_path = "predictions/TU_Veinseg224/TU_pretrain_R50-ViT-B_16_skip3_bs4_224/ICA2_pred.nii.gz"
    visualize(img_path, gt_path, pred_path)
