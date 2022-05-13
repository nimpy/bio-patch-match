import matplotlib.pyplot as plt
import os
import numpy as np

plots_dir = "images/plots"


def plot_patch_diffs(patch_diffs, filename):
    plt.figure(figsize=(10, 8))
    plt.plot(patch_diffs[:-1])  # the last one is the one that's been added in the beginning artificially
    plt.savefig(os.path.join(plots_dir, filename))
    plt.show()


def visualise_segmentation(original, ground_truth, prediction, alpha=0.35):
    assert original.shape == ground_truth.shape, "The shapes of images are not the same."
    assert original.shape == prediction.shape, "The shapes of images are not the same."

    print(original.shape)

    # TODO check the type of arrays and whether it is in [0..1] or [0..255]
    #      (For the moment, I assume np.float64 and [0..1])
    # TODO generalise s.t. original isn't necessarily greyscale

    img_vis = np.zeros((original.shape[0], original.shape[1], 3), dtype=np.float64)

    for i in range(original.shape[0]):
        for j in range(original.shape[1]):
            if (ground_truth[i, j] >= 0.5) and (prediction[i, j] >= 0.5):  # TODO optimise this
                img_vis[i, j, 0] = original[i, j]
                img_vis[i, j, 1] = original[i, j]
                img_vis[i, j, 2] = original[i, j]
            elif (ground_truth[i, j] < 0.5) and (prediction[i, j] < 0.5):  # blue for TP
                img_vis[i, j, 0] = original[i, j] * (1.0 - alpha) + 0. * alpha
                img_vis[i, j, 1] = original[i, j] * (1.0 - alpha) + 0. * alpha
                img_vis[i, j, 2] = original[i, j] * (1.0 - alpha) + 1. * alpha
            elif (ground_truth[i, j] < 0.5) and (prediction[i, j] >= 0.5):  # red for FN
                img_vis[i, j, 0] = original[i, j] * (1.0 - alpha) + 1. * alpha
                img_vis[i, j, 1] = original[i, j] * (1.0 - alpha) + 0. * alpha
                img_vis[i, j, 2] = original[i, j] * (1.0 - alpha) + 0. * alpha
            elif (ground_truth[i, j] >= 0.5) and (prediction[i, j], 0.5):  # orange for FP
                img_vis[i, j, 0] = original[i, j] * (1.0 - alpha) + 1. * alpha
                img_vis[i, j, 1] = original[i, j] * (1.0 - alpha) + 0.5 * alpha
                img_vis[i, j, 2] = original[i, j] * (1.0 - alpha) + 0. * alpha

    return img_vis

