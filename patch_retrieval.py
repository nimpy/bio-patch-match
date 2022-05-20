from descriptors_encoding import compute_descriptor, calculate_diff
import numpy as np
from sklearn.metrics import recall_score


EPS = 0.0001


def insert_sorted_diffs_coords_and_positions(diff, x_compare, y_compare, position, patches_diffs,
                                             patches_x_coords, patches_y_coords, patches_positions):
    # sorting
    for i in range(len(patches_diffs)):
        if diff < patches_diffs[i]:
            patches_diffs.insert(i, diff)
            patches_x_coords.insert(i, x_compare)
            patches_y_coords.insert(i, y_compare)
            patches_positions.insert(i, position)
            return patches_diffs, patches_x_coords, patches_y_coords, patches_positions


def retrieve_patches_sorted_by_diff(query_patch, image, descriptor, patch_size, compare_stride):
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

            diff = calculate_diff(query_patch_descr, compare_patch_descr)

            if diff < EPS:  # when using VAE check it's not the same patch
                counter_compare_patches += 1
                continue

            patches_diffs, patches_x_coords, patches_y_coords, patches_positions = \
                insert_sorted_diffs_coords_and_positions(diff, x_compare, y_compare, counter_compare_patches,
                                                         patches_diffs, patches_x_coords, patches_y_coords,
                                                         patches_positions)
            counter_compare_patches += 1

    return patches_diffs, patches_x_coords, patches_y_coords, patches_positions


def prune_closest_n_patches(nr_closest_patches, patches_x_coords, patches_y_coords, patches_positions):
    results_patches_x_coords = patches_x_coords[:nr_closest_patches]
    results_patches_y_coords = patches_y_coords[:nr_closest_patches]
    results_patches_positions = patches_positions[:nr_closest_patches]
    return results_patches_x_coords, results_patches_y_coords, results_patches_positions


def retrieve_patch_matches_for_2queries_sum_diffs(query_patch1, query_patch2, image, descriptor, patch_size, compare_stride):
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

            diff1 = calculate_diff(query_patch_descr1, compare_patch_descr)
            diff2 = calculate_diff(query_patch_descr2, compare_patch_descr)

            if diff1 < EPS or diff2 < EPS:  # when using VAE check it's not the same patch
                counter_compare_patches += 1
                continue

            diff = diff1 + diff2

            patches_diffs, patches_x_coords, patches_y_coords, patches_positions = \
                insert_sorted_diffs_coords_and_positions(diff, x_compare, y_compare, counter_compare_patches,
                                                         patches_diffs, patches_x_coords, patches_y_coords,
                                                         patches_positions)

            counter_compare_patches += 1

    return patches_diffs, patches_x_coords, patches_y_coords, patches_positions


def calculate_percentage_correctly_labelled_pixels(label, positive_label_value, patch_size,
                                                   retrieved_patches_positions, retrieved_patches_x_coords,
                                                   retrieved_patches_y_coords):
    retrieved_patches_correct_pixels = []
    completely_missed_count = 0
    for i, patch_match in enumerate(retrieved_patches_positions):
        patch_match_label = label[retrieved_patches_x_coords[i]: retrieved_patches_x_coords[i] + patch_size,
                                  retrieved_patches_y_coords[i]: retrieved_patches_y_coords[i] + patch_size]
        correct_pixels = np.sum(patch_match_label == positive_label_value)  # TODO check this!!
        retrieved_patches_correct_pixels.append(correct_pixels / (patch_size ** 2))
        if correct_pixels == 0:
            completely_missed_count += 1
    mean_correct_pixels = np.array(retrieved_patches_correct_pixels).mean()
    return completely_missed_count, mean_correct_pixels


def calculate_sensitivity(label, positive_label_value, patch_size, retrieved_patches_positions,
                          retrieved_patches_x_coords, retrieved_patches_y_coords):

    # TODO extract this to be a method, from here...
    label = label / 255.0
    if positive_label_value == 0:
        y_pred = np.ones_like(label)
    elif positive_label_value == 1:
        y_pred = np.zeros_like(label)
    else:
        raise ValueError("Positive label value is strange -- it should be 0 or 1 but it is", positive_label_value)
    for x, y in zip(retrieved_patches_x_coords, retrieved_patches_y_coords):
        y_pred[x: x + patch_size, y: y + patch_size] = positive_label_value
    # TODO ...until here

    return recall_score(np.array(label, dtype=np.uint8).flatten(), np.array(y_pred, dtype=np.uint8).flatten(),
                        pos_label=positive_label_value)


# TODO calculate also f1 score!
def calculate_retrieval_score(label, patch_size, retrieved_patches_positions,
                              retrieved_patches_x_coords, retrieved_patches_y_coords, positive_label_value=0):
    # return calculate_percentage_correctly_labelled_pixels(label, positive_label_value, patch_size,
    #                                                       retrieved_patches_positions, retrieved_patches_x_coords,
    #                                                       retrieved_patches_y_coords)[1]  # TODO change final evaluation
    return calculate_sensitivity(label, positive_label_value, patch_size, retrieved_patches_positions,
                                 retrieved_patches_x_coords, retrieved_patches_y_coords)
