import os
from pathlib import Path
import torch
import imageio
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches

import models.ae as ae

weights_dir = 'weights/'
weights_path = os.path.join(weights_dir, 'ae_best.pth.tar')

data_dir = 'images/'
image_filename = 'data_0185.png'  # 0267
label_filename = 'data_0185_label.png'  # 0267
query_patch_filename1 = 'data_0165_mito_crop.png'
query_patch_filename2 = 'data_0267_mito_crop.png'

image_path = os.path.join(data_dir, image_filename)
label_path = os.path.join(data_dir, label_filename)
query_patch_path1 = os.path.join(data_dir, query_patch_filename1)
query_patch_path2 = os.path.join(data_dir, query_patch_filename2)

result_visualisation_dir = 'images/res_vis_' + Path(image_filename).stem + "_" + Path(query_patch_path1).stem  + "_" + Path(query_patch_path2).stem
result_visualisation_labels_dir = 'images/res_vis_lab_' + Path(image_filename).stem + "_" + Path(query_patch_path1).stem + "_" + Path(query_patch_path2).stem


EPS = 0.0001
patch_size = 65
compare_stride = 8
nr_similar_patches = 16


def load_images():
    image = imageio.imread(image_path)
    label = imageio.imread(label_path)
    query_patch1 = imageio.imread(query_patch_path1)
    query_patch2 = imageio.imread(query_patch_path2)
    return image, label, query_patch1, query_patch2


def load_descriptors():
    model = ae.AE(32)
    model.load_state_dict(torch.load(weights_path)['state_dict'])
    model.eval()
    # model = model.cuda()  # TODO make it use GPU
    return model


def calculate_ssd(img1, img2):
    """Computing the sum of squared differences (SSD) between two images."""
    if img1.shape != img2.shape:
        raise Exception("Images don't have the same shape: ", img1.shape, "and", img2.shape)
    return np.sum((np.array(img1, dtype=np.float32) - np.array(img2, dtype=np.float32))**2)


def compute_descriptor(descr, patch):
    variational = False  # isinstance(descr, vae.BetaVAE) or isinstance(descr, vae_ir.BetaVAE)
    patch = np.array(patch)
    patch = patch / 255.0
    patch = np.expand_dims(np.expand_dims(patch, axis=0), axis=0)
    patch = torch.from_numpy(patch).float()
    if variational:
        patch_encoding, _, _ = descr.encode(patch)
    else:
        patch_encoding = descr.encode(patch)
    patch_encoding = patch_encoding.detach().numpy()
    patch_encoding = patch_encoding.reshape(patch_encoding.shape[0], np.product(patch_encoding.shape[1:]))
    return patch_encoding[0]


def retrieve_patch_matches(query_patch, image, descriptor, patch_size, compare_stride):
    image_height = image.shape[0]
    image_width = image.shape[1]

    query_patch_descr = compute_descriptor(descriptor, query_patch)

    counter_compare_patches = 0

    patches_diffs = [1000000000]
    patches_x_coords = [-1]
    patches_y_coords = [-1]
    patches_positions = [-1]

    for y_compare in range(0, image_width - patch_size + 1, compare_stride):
        for x_compare in range(0, image_height - patch_size + 1, compare_stride):

            compare_patch = image[x_compare: x_compare + patch_size, y_compare: y_compare + patch_size]

            compare_patch_descr = compute_descriptor(descriptor, compare_patch)

            diff = calculate_ssd(query_patch_descr, compare_patch_descr)

            if diff < EPS:  # when using VAE check it's not the same patch
                counter_compare_patches += 1
                continue

            # sorting
            for i in range(len(patches_diffs)):
                if diff < patches_diffs[i]:
                    patches_diffs.insert(i, diff)
                    patches_x_coords.insert(i, x_compare)
                    patches_y_coords.insert(i, y_compare)
                    patches_positions.insert(i, counter_compare_patches)
                    break

            counter_compare_patches += 1

    return patches_diffs, patches_x_coords, patches_y_coords, patches_positions


def retrieve_patch_matches_for_2queries(query_patch1, query_patch2, image, descriptor, patch_size, compare_stride):
    image_height = image.shape[0]
    image_width = image.shape[1]

    query_patch_descr1 = compute_descriptor(descriptor, query_patch1)
    query_patch_descr2 = compute_descriptor(descriptor, query_patch2)

    counter_compare_patches = 0

    patches_diffs = [1000000000]
    patches_x_coords = [-1]
    patches_y_coords = [-1]
    patches_positions = [-1]

    for y_compare in range(0, image_width - patch_size + 1, compare_stride):
        for x_compare in range(0, image_height - patch_size + 1, compare_stride):

            compare_patch = image[x_compare: x_compare + patch_size, y_compare: y_compare + patch_size]

            compare_patch_descr = compute_descriptor(descriptor, compare_patch)

            diff1 = calculate_ssd(query_patch_descr1, compare_patch_descr)
            diff2 = calculate_ssd(query_patch_descr2, compare_patch_descr)

            if diff1 < EPS or diff2 < EPS:  # when using VAE check it's not the same patch
                counter_compare_patches += 1
                continue

            diff = diff1 + diff2

            # sorting
            for i in range(len(patches_diffs)):
                if diff < patches_diffs[i]:
                    patches_diffs.insert(i, diff)
                    patches_x_coords.insert(i, x_compare)
                    patches_y_coords.insert(i, y_compare)
                    patches_positions.insert(i, counter_compare_patches)
                    break

            counter_compare_patches += 1

    return patches_diffs, patches_x_coords, patches_y_coords, patches_positions


def plot_patch_matches_and_metrics_for_different_nr_similar_patches(nr_similar_patches_list, image, label,
                                                                patches_x_coords, patches_y_coords, patches_positions):
    maximum_correct_pixels = patch_size ** 2

    for nr_similar_patches in nr_similar_patches_list:

        print(nr_similar_patches)

        results_patches_x_coords = patches_x_coords[:nr_similar_patches]
        results_patches_y_coords = patches_y_coords[:nr_similar_patches]
        results_patches_positions = patches_positions[:nr_similar_patches]

        # calculating the percentage of correctly labelled pixels and the number of completely failed matches
        results_patches_correct_pixels = []
        failed_count = 0
        for i, patch_match in enumerate(results_patches_positions):
            patch_match_label = label[results_patches_x_coords[i]: results_patches_x_coords[i] + patch_size,
                                results_patches_y_coords[i]: results_patches_y_coords[i] + patch_size]
            correct_pixels = np.sum(patch_match_label == 1)  # assumes label 1 for the class of interest TODO generalise
            results_patches_correct_pixels.append(correct_pixels / maximum_correct_pixels)
            if correct_pixels == 0:
                failed_count += 1
        mean_correct_pixels = np.array(results_patches_correct_pixels).mean()

        # plotting
        fig, ax = plt.subplots(1)
        fig.set_size_inches(18.5, 10.5)
        fig.set_dpi(100)
        ax.imshow(image, cmap='gray')

        for i, patch_match in enumerate(results_patches_positions):
            rect = patches.Rectangle((results_patches_y_coords[i], results_patches_x_coords[i]),
                                     65, 65, linewidth=1, edgecolor='r', facecolor='none')
            ax.add_patch(rect)

        # saving result visualisation
        result_visualisation_file_name = Path(image_filename).stem + '_patchmatch' + str(nr_similar_patches).zfill(3) + '_correctpixels' + \
            "{:.2f}".format(mean_correct_pixels * 100) + '_failed' + str(failed_count) + '.png'

        Path(result_visualisation_dir).mkdir(parents=True, exist_ok=True)
        plt.savefig(os.path.join(result_visualisation_dir, result_visualisation_file_name), bbox_inches='tight')
        plt.show()
        print(mean_correct_pixels * 100)
        print('failed:', failed_count)

        print('\n\n')


def plot_patch_matches_and_metrics_for_different_nr_similar_patches2_AND(nr_similar_patches_list, image, label,
                                                                patches_x_coords, patches_y_coords, patches_positions):
    maximum_correct_pixels = patch_size ** 2

    for nr_similar_patches in nr_similar_patches_list:

        print(nr_similar_patches)

        results_patches_x_coords = patches_x_coords[:nr_similar_patches]
        results_patches_y_coords = patches_y_coords[:nr_similar_patches]
        results_patches_positions = patches_positions[:nr_similar_patches]

        prediction_image = np.zeros(image.shape, dtype=np.uint8)

        # calculating the percentage of correctly labelled pixels and the number of completely failed matches
        results_patches_correct_pixels = []
        failed_count = 0
        for i, patch_match in enumerate(results_patches_positions):
            patch_match_label = label[results_patches_x_coords[i]: results_patches_x_coords[i] + patch_size,
                                results_patches_y_coords[i]: results_patches_y_coords[i] + patch_size]
            correct_pixels = np.sum(patch_match_label == 1)  # assumes label 1 for the class of interest TODO generalise
            results_patches_correct_pixels.append(correct_pixels / maximum_correct_pixels)
            if correct_pixels == 0:
                failed_count += 1

            prediction_image[results_patches_x_coords[i]: results_patches_x_coords[i] + patch_size,
                                results_patches_y_coords[i]: results_patches_y_coords[i] + patch_size] = 255
        mean_correct_pixels = np.array(results_patches_correct_pixels).mean()


        prediction_visualisation = visualise_segmentation(image / 255.0, label / 255.0, 1 - (prediction_image / 255.0))

        # saving result visualisation
        result_visualisation_file_name = Path(image_filename).stem + '_patchmatch' + str(nr_similar_patches).zfill(3) + '_correctpixels' + \
            "{:.2f}".format(mean_correct_pixels * 100) + '_failed' + str(failed_count) + '_labels.png'

        Path(result_visualisation_labels_dir).mkdir(parents=True, exist_ok=True)

        # plotting
        fig, ax = plt.subplots(1)
        fig.set_size_inches(18.5, 10.5)
        fig.set_dpi(100)
        ax.imshow(prediction_visualisation)
        plt.savefig(os.path.join(result_visualisation_labels_dir, result_visualisation_file_name), bbox_inches='tight')
        plt.show()
        print(mean_correct_pixels * 100)
        print('failed:', failed_count)

        print('\n\n')


def plot_patch_matches_and_metrics_for_different_nr_similar_patches2_OR(nr_similar_patches_list, image, label,
                                                    patches_x_coords1, patches_y_coords1, patches_positions1,
                                                    patches_x_coords2, patches_y_coords2, patches_positions2):

    image_height = image.shape[0]
    image_width = image.shape[1]

    maximum_correct_pixels = patch_size ** 2

    for nr_similar_patches in nr_similar_patches_list:

        print(nr_similar_patches)

        results_patches_x_coords1 = patches_x_coords1[:nr_similar_patches]
        results_patches_y_coords1 = patches_y_coords1[:nr_similar_patches]
        # results_patches_positions1 = patches_positions1[:nr_similar_patches]

        results_patches_x_coords2 = patches_x_coords2[:nr_similar_patches]
        results_patches_y_coords2 = patches_y_coords2[:nr_similar_patches]
        # results_patches_positions2 = patches_positions2[:nr_similar_patches]

        prediction_image = np.zeros(image.shape, dtype=np.uint8)

        # calculating the percentage of correctly labelled pixels and the number of completely failed matches
        results_patches_correct_pixels = []
        failed_count = 0

        # for i, patch_match in enumerate(results_patches_positions):
        for y_compare in range(0, image_width - patch_size + 1, compare_stride):
            for x_compare in range(0, image_height - patch_size + 1, compare_stride):

                if y_compare in results_patches_y_coords1 and y_compare in results_patches_y_coords2 and x_compare in results_patches_x_coords1 and x_compare in results_patches_x_coords2:
                    if results_patches_y_coords1.index(y_compare) == results_patches_x_coords1.index(x_compare) and results_patches_y_coords2.index(y_compare) == results_patches_x_coords2.index(x_compare):

                        patch_match_label = label[x_compare: x_compare + patch_size, y_compare: y_compare + patch_size]
                        correct_pixels = np.sum(patch_match_label == 1)  # assumes label 1 for the class of interest TODO generalise
                        results_patches_correct_pixels.append(correct_pixels / maximum_correct_pixels)
                        if correct_pixels == 0:
                            failed_count += 1

                        prediction_image[x_compare: x_compare + patch_size, y_compare: y_compare + patch_size] = 255

        mean_correct_pixels = np.array(results_patches_correct_pixels).mean()

        prediction_visualisation = visualise_segmentation(image / 255.0, label / 255.0, 1 - (prediction_image / 255.0))

        # saving result visualisation
        result_visualisation_file_name = Path(image_filename).stem + '_patchmatch' + str(nr_similar_patches).zfill(3) + '_correctpixels' + \
            "{:.2f}".format(mean_correct_pixels * 100) + '_failed' + str(failed_count) + '_labels.png'

        Path(result_visualisation_labels_dir + '_OR').mkdir(parents=True, exist_ok=True)

        # plotting
        fig, ax = plt.subplots(1)
        fig.set_size_inches(18.5, 10.5)
        fig.set_dpi(100)
        ax.imshow(prediction_visualisation)
        plt.savefig(os.path.join(result_visualisation_labels_dir + '_OR', result_visualisation_file_name), bbox_inches='tight')
        plt.show()
        print(mean_correct_pixels * 100)
        print('failed:', failed_count)

        print('\n\n')

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
            if (ground_truth[i, j] >= 0.5) and (prediction[i, j] >= 0.5):
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


if __name__ == '__main__':
    image, label, query_patch1, query_patch2 = load_images()
    descriptor = load_descriptors()

    patches_diffs1, patches_x_coords1, patches_y_coords1, patches_positions1 = retrieve_patch_matches(query_patch1, image,
                                                                    descriptor, patch_size, compare_stride)

    patches_diffs2, patches_x_coords2, patches_y_coords2, patches_positions2 = retrieve_patch_matches(query_patch1, image,
                                                                    descriptor, patch_size, compare_stride)

    # patches_diffs, patches_x_coords, patches_y_coords, patches_positions = retrieve_patch_matches_for_2queries(query_patch1,
    #                                                         query_patch2, image, descriptor, patch_size, compare_stride)

    nr_similar_patches_list = [i * 10 + 6 for i in range(12)]  # zum Beispiel

    plot_patch_matches_and_metrics_for_different_nr_similar_patches2_OR(nr_similar_patches_list, image, label,
                                                    patches_x_coords1, patches_y_coords1, patches_positions1,
                                                    patches_x_coords2, patches_y_coords2, patches_positions2)

    print()