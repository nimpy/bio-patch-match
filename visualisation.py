import matplotlib.pyplot as plt
import matplotlib.patches as patches
import os
from pathlib import Path
import numpy as np

plots_dir = "images/plots"
PLOT_HEIGHT = 10.5
PLOT_WIDTH = 18.5
PLOT_DPI = 100


def plot_patch_diffs(patch_diffs, filename):
    plt.figure(figsize=(PLOT_WIDTH, PLOT_HEIGHT))
    plt.plot(patch_diffs[:-1])  # the last one is the one that's been added in the beginning artificially
    plt.savefig(os.path.join(plots_dir, filename))
    plt.show()


def visualise_as_segmentation(original, ground_truth, prediction, alpha=0.35):
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


def visualise_retrieved_patches_as_rectangles(image, patch_size, result_visualisation_dir, visualisation_file_name,
                                              results_patches_positions, results_patches_x_coords,
                                              results_patches_y_coords):
    fig, ax = plt.subplots(1)
    fig.set_size_inches(PLOT_WIDTH, PLOT_HEIGHT)
    fig.set_dpi(PLOT_DPI)
    ax.imshow(image, cmap='gray')
    for i, patch_match in enumerate(results_patches_positions):
        rect = patches.Rectangle((results_patches_y_coords[i], results_patches_x_coords[i]),
                                 patch_size, patch_size, linewidth=1, edgecolor='r', facecolor='none')
        ax.add_patch(rect)

    # saving result visualisation
    Path(result_visualisation_dir).mkdir(parents=True, exist_ok=True)
    plt.savefig(os.path.join(result_visualisation_dir, visualisation_file_name), bbox_inches='tight')
    plt.show()


def visualise_retrieved_patches_as_segmentation(image, label, patch_size, result_visualisation_dir, visualisation_file_name,
                                                results_patches_positions, results_patches_x_coords,
                                                results_patches_y_coords):
    prediction_image = np.zeros(image.shape, dtype=np.uint8)
    for i, patch_match in enumerate(results_patches_positions):
        prediction_image[results_patches_x_coords[i]: results_patches_x_coords[i] + patch_size,
                         results_patches_y_coords[i]: results_patches_y_coords[i] + patch_size] = 255

    prediction_visualisation = visualise_as_segmentation(image / 255.0, label / 255.0, 1 - (prediction_image / 255.0))
    fig, ax = plt.subplots(1)
    fig.set_size_inches(PLOT_WIDTH, PLOT_HEIGHT)
    fig.set_dpi(PLOT_DPI)
    ax.imshow(prediction_visualisation)
    plt.savefig(os.path.join(result_visualisation_dir, visualisation_file_name), bbox_inches='tight')
    plt.show()
