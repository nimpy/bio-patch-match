import os
from pathlib import Path

import imageio
import numpy as np

from patch_retrieval import retrieve_patches_sorted_by_diff, prune_closest_n_patches, calculate_retrieval_score
from descriptors_encoding import load_descriptors

def is_score_better_than_best(current_score, best_score, maximise_score):
    if maximise_score:
        return current_score > best_score
    else:
        return current_score < best_score


def find_best_threshold(query_patches, images, labels, thresholds, descriptor, patch_size, compare_stride,
                        maximise_score=True):

    if maximise_score:  # TODO change default best_score values
        best_threshold_score = 0
    else:
        best_threshold_score = 1000000
    best_threshold = -1

    for threshold in thresholds:
        scores = []
        for query_patch in query_patches:
            for image, label in zip(images, labels):

                patches_diffs, patches_x_coords, patches_y_coords, patches_positions = \
                    retrieve_patches_sorted_by_diff(query_patch, image, descriptor, patch_size, compare_stride)

                nr_similar_patches = threshold  # TODO calculate in some smarter way?
                retrieved_patches_x_coords, retrieved_patches_y_coords, retrieved_patches_positions = \
                    prune_closest_n_patches(nr_similar_patches, patches_x_coords, patches_y_coords, patches_positions)

                score = calculate_retrieval_score(label, patch_size, retrieved_patches_positions,
                                                  retrieved_patches_x_coords, retrieved_patches_y_coords)
                print(score)
                scores.append(score)
        threshold_score = np.mean(np.array(scores))
        print("For threshold", threshold, "the score is", threshold_score)
        if is_score_better_than_best(threshold_score, best_threshold_score, maximise_score):
            best_threshold_score = threshold_score
            best_threshold = threshold
            print("   Found the best threshold so far:", best_threshold, "with the score: ", best_threshold_score)

    return best_threshold


if __name__ == '__main__':
    base_data_dir = 'data/'
    images_dir = 'images'
    labels_dir = 'labels'
    query_patches_dir = 'query_patches'

    image_filenames = os.listdir(os.path.join(base_data_dir, images_dir))
    label_filenames = os.listdir(os.path.join(base_data_dir, labels_dir))
    query_patch_filenames = os.listdir(os.path.join(base_data_dir, query_patches_dir))

    image_filenames.sort()
    label_filenames.sort()
    query_patch_filenames.sort()

    images = []
    for image_filename in image_filenames:
        images.append(imageio.imread(os.path.join(base_data_dir, images_dir, image_filename)))
    labels = []
    for label_filename in label_filenames:
        labels.append(imageio.imread(os.path.join(base_data_dir, labels_dir, label_filename)))
    query_patches = []
    for query_patch_filename in query_patch_filenames:
        query_patches.append(imageio.imread(os.path.join(base_data_dir, query_patches_dir, query_patch_filename)))

    thresholds = [6, 26, 46, 66]

    descriptor = load_descriptors()
    patch_size = 65
    compare_stride = 8

    find_best_threshold(query_patches, images, labels, thresholds, descriptor, patch_size, compare_stride)