from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE

import os
import imageio
import numpy as np
import matplotlib.pyplot as plt
import datetime
import time

from descriptors_encoding import load_descriptors, compute_descriptor


patch_size = 65
compare_stride = 8
mitochondria_threshold = 0.2


def load_images_and_labels():
    data_dir = 'data/'
    images_dir = 'images/'
    labels_dir = 'labels/'

    images_filenames = os.listdir(os.path.join(data_dir, images_dir))
    labels_filenames = os.listdir(os.path.join(data_dir, labels_dir))

    images_filenames.sort()
    labels_filenames.sort()

    images = []
    labels = []

    for image_filename, label_filename in zip(images_filenames, labels_filenames):
        images.append(imageio.imread(os.path.join(data_dir, images_dir, image_filename)))
        labels.append(imageio.imread(os.path.join(data_dir, labels_dir, label_filename)))
    return images, labels


if __name__ == '__main__':
    descriptor = load_descriptors()

    images, labels = load_images_and_labels()

    image_height = images[0].shape[0]
    image_width = images[0].shape[1]

    total_patches_count = len(range(0, image_width - patch_size + 1, compare_stride)) * \
                    len(range(0, image_height - patch_size + 1, compare_stride))
    total_patches_count *= len(images)  # because we will take the patches for every image

    patch_descrs = np.zeros((total_patches_count, 32))  # TODO make not hard-coded
    patch_labels = []
    patch_count = 0

    for image, label in zip(images, labels):
        for y in range(0, image_width - patch_size + 1, compare_stride):
            for x in range(0, image_height - patch_size + 1, compare_stride):

                patch = image[x: x + patch_size, y: y + patch_size]
                patch_descr = compute_descriptor(descriptor, patch)
                patch_descrs[patch_count] = patch_descr

                patch_label = label[x: x + patch_size, y: y + patch_size]
                mitochondria_pixels = np.sum(patch_label == 1)  # 1 for mito and 255 non-mito
                if (mitochondria_pixels / (patch_size ** 2)) > mitochondria_threshold:
                    patch_labels.append(1)  # (mostly) mitochondria
                else:
                    patch_labels.append(0)  # (mostly) not mitochondria

                patch_count += 1

    # print(patch_descrs.shape)
    # kmeans = KMeans(n_clusters=5, random_state=0).fit(patch_descrs)
    # print(kmeans.labels_)

    patch_labels = np.array(patch_labels)

    # # 2D visualisation
    # pca = PCA(n_components=2)
    # patch_descrs_transformed = pca.fit_transform(patch_descrs)
    # plt.scatter(patch_descrs_transformed[:, 0], patch_descrs_transformed[:, 1], c=patch_labels)
    # plt.savefig('images/plots/mitochondria_' + str(mitochondria_threshold) + '_2D-PCA__' + datetime.datetime.now().strftime("%Y%m%d_%H%M%S") + '.jpg')
    # plt.show()
    # #
    # time.sleep(1)
    # #
    # # 3D visualisation
    # pca = PCA(n_components=3)
    # patch_descrs_transformed = pca.fit_transform(patch_descrs)
    # fig = plt.figure(figsize=(12, 12))
    # ax = fig.add_subplot(projection='3d')
    # ax.scatter(patch_descrs_transformed[:, 0], patch_descrs_transformed[:, 1], patch_descrs_transformed[:, 2], c=patch_labels)
    # plt.savefig('images/plots/mitochondria_' + str(mitochondria_threshold) + '_3D-PCA__' + datetime.datetime.now().strftime("%Y%m%d_%H%M%S") + '.jpg')
    # plt.show()

    # t-SNE
    patch_descrs_tSNE = TSNE(n_components=3, learning_rate='auto', init='random').fit_transform(patch_descrs)
    fig = plt.figure(figsize=(12, 12))
    ax = fig.add_subplot(projection='3d')
    ax.scatter(patch_descrs_tSNE[:, 0], patch_descrs_tSNE[:, 1], patch_descrs_tSNE[:, 2], c=patch_labels)
    plt.savefig('images/plots/mitochondria_' + str(mitochondria_threshold) + '_tSNE__' + datetime.datetime.now().strftime("%Y%m%d_%H%M%S") + '.jpg')
    plt.show()
