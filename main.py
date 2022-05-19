import os
from pathlib import Path

import imageio
import numpy as np
import matplotlib.pyplot as plt

from descriptors_encoding import load_descriptors, compute_descriptor, calculate_diff
from patch_retrieval import retrieve_patches_sorted_by_diff, prune_closest_n_patches, calculate_retrieval_score
from patch_retrieval import retrieve_patch_matches_for_2queries_sum_diffs
from visualisation import plot_patch_diffs, visualise_as_segmentation
from visualisation import visualise_retrieved_patches_as_rectangles, visualise_retrieved_patches_as_segmentation

data_dir = 'images/'
image_filename = 'data_0267.png'  # 0267
label_filename = 'data_0267_label.png'  # 0267
query_patch_filename1 = 'data_0165_mito_crop.png'
query_patch_filename2 = 'data_0267_mito_negative_crop2.png'  # data_0267_mito_negative_crop2  data_0209_mito_crop3

image_path = os.path.join(data_dir, image_filename)
label_path = os.path.join(data_dir, label_filename)
query_patch_path1 = os.path.join(data_dir, query_patch_filename1)
query_patch_path2 = os.path.join(data_dir, query_patch_filename2)

result_visualisation_dir = 'images/res_vis_' + Path(image_filename).stem + "_" + Path(query_patch_path1).stem + "_" + Path(query_patch_path2).stem
result_visualisation_labels_dir = 'images/res_vis_lab_' + Path(image_filename).stem + "_" + Path(query_patch_path1).stem + "_" + Path(query_patch_path2).stem


patch_size = 65
compare_stride = 8
nr_similar_patches = 16
positive_label_value = 1


def load_images():
    image = imageio.imread(image_path)
    label = imageio.imread(label_path)
    query_patch1 = imageio.imread(query_patch_path1)
    query_patch2 = imageio.imread(query_patch_path2)
    return image, label, query_patch1, query_patch2


def plot_patch_matches_and_metrics_for_different_nr_similar_patches(nr_similar_patches_list, image, label,
                                                                patches_x_coords, patches_y_coords, patches_positions):

    for nr_similar_patches in nr_similar_patches_list:

        print(nr_similar_patches)

        # TODO rename vars: results_ to retrieved_
        results_patches_x_coords, results_patches_y_coords, results_patches_positions = \
            prune_closest_n_patches(nr_similar_patches, patches_x_coords, patches_y_coords, patches_positions)

        # calculating the percentage of correctly labelled pixels and the number of completely failed matches
        failed_count, mean_correct_pixels = \
            calculate_retrieval_score(label, positive_label_value, patch_size, results_patches_positions,
                                                           results_patches_x_coords, results_patches_y_coords)

        result_visualisation_file_name = Path(image_filename).stem + '_patchmatch' + str(nr_similar_patches).zfill(
            3) + '_correctpixels' + "{:.2f}".format(mean_correct_pixels * 100) + '_failed' + str(failed_count) + '.png'

        # plotting
        visualise_retrieved_patches_as_rectangles(image, patch_size, result_visualisation_dir,
                                                  result_visualisation_file_name, results_patches_positions,
                                                  results_patches_x_coords, results_patches_y_coords)
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

        # saving result visualisation
        result_visualisation_file_name = Path(image_filename).stem + '_patchmatch' + str(nr_similar_patches).zfill(3) + '_correctpixels' + \
            "{:.2f}".format(mean_correct_pixels * 100) + '_failed' + str(failed_count) + '_labels.png'

        Path(result_visualisation_labels_dir).mkdir(parents=True, exist_ok=True)

        # plotting
        visualise_retrieved_patches_as_segmentation(image, label, patch_size, result_visualisation_dir,
                                                    result_visualisation_file_name,
                                                    results_patches_positions, results_patches_x_coords,
                                                    results_patches_y_coords)
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

        prediction_visualisation = visualise_as_segmentation(image / 255.0, label / 255.0, 1 - (prediction_image / 255.0))

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


def plot_patch_matches_and_metrics_for_different_nr_similar_patches2_1ANDNOT2(nr_similar_patches_list, image, label,
                                                    patches_x_coords1, patches_y_coords1, patches_positions1,
                                                    patches_x_coords2, patches_y_coords2, patches_positions2):

    image_height = image.shape[0]
    image_width = image.shape[1]

    maximum_correct_pixels = patch_size ** 2

    for nr_similar_patches in nr_similar_patches_list:

        nr_dissimilar_patches = 5000

        print(nr_similar_patches, nr_dissimilar_patches)

        results_patches_x_coords1 = patches_x_coords1[:nr_similar_patches]
        results_patches_y_coords1 = patches_y_coords1[:nr_similar_patches]
        # results_patches_positions1 = patches_positions1[:nr_similar_patches]

        results_patches_x_coords2 = patches_x_coords2[-nr_dissimilar_patches:]
        results_patches_y_coords2 = patches_y_coords2[-nr_dissimilar_patches:]
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

        prediction_visualisation = visualise_as_segmentation(image / 255.0, label / 255.0, 1 - (prediction_image / 255.0))

        # saving result visualisation
        result_visualisation_file_name = Path(image_filename).stem + '_patchmatch' + str(nr_similar_patches).zfill(3) + '_correctpixels' + \
            "{:.2f}".format(mean_correct_pixels * 100) + '_failed' + str(failed_count) + '_labels.png'

        Path(result_visualisation_labels_dir + '_1ANDNOT2').mkdir(parents=True, exist_ok=True)

        # plotting
        fig, ax = plt.subplots(1)
        fig.set_size_inches(18.5, 10.5)
        fig.set_dpi(100)
        ax.imshow(prediction_visualisation)
        plt.savefig(os.path.join(result_visualisation_labels_dir + '_1ANDNOT2', result_visualisation_file_name), bbox_inches='tight')
        plt.show()
        print(mean_correct_pixels * 100)
        print('failed:', failed_count)

        print('\n\n')


if __name__ == '__main__':
    image, label, query_patch1, query_patch2 = load_images()
    descriptor = load_descriptors()

    # patches_diffs1, patches_x_coords1, patches_y_coords1, patches_positions1 = retrieve_patches_sorted_by_diff(query_patch1, image,
    #                                                                                                            descriptor, patch_size, compare_stride)
    #
    # patches_diffs2, patches_x_coords2, patches_y_coords2, patches_positions2 = retrieve_patches_sorted_by_diff(query_patch1, image,
    #                                                                                                            descriptor, patch_size, compare_stride)
    #
    # plot_patch_diffs(patches_diffs1, "diffs__" + os.path.splitext(image_filename)[0] + "__" + os.path.splitext(query_patch_filename1)[0] + ".png")
    # TODO diffs2

    nr_similar_patches_list = [i * 10 + 6 for i in range(12)]  # zum Beispiel

    # plot_patch_matches_and_metrics_for_different_nr_similar_patches2_OR(nr_similar_patches_list, image, label,
    #                                                 patches_x_coords1, patches_y_coords1, patches_positions1,
    #                                                 patches_x_coords2, patches_y_coords2, patches_positions2)

    # plot_patch_matches_and_metrics_for_different_nr_similar_patches2_1ANDNOT2(nr_similar_patches_list, image, label,
    #                                                 patches_x_coords1, patches_y_coords1, patches_positions1,
    #                                                 patches_x_coords2, patches_y_coords2, patches_positions2)


    patches_diffs, patches_x_coords, patches_y_coords, patches_positions = retrieve_patch_matches_for_2queries_sum_diffs(query_patch1,
                                                            query_patch2, image, descriptor, patch_size, compare_stride)
    plot_patch_matches_and_metrics_for_different_nr_similar_patches2_AND(nr_similar_patches_list, image, label,
                                                                patches_x_coords, patches_y_coords, patches_positions)


    print()
