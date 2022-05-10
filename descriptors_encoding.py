import torch
import os
import numpy as np

import models.ae as ae


weights_dir = 'weights/'
weights_path = os.path.join(weights_dir, 'ae_best.pth.tar')


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

