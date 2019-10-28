import matplotlib.pyplot as plt
import os

import numpy as np

image_output_dir = "image_processed"
os.makedirs(image_output_dir, exist_ok=True)


def save_im(imname, im, cmap=None):
    impath = os.path.join(image_output_dir, imname)
    plt.imsave(impath, im, cmap=cmap)


def greyscale(im):
    """ Converts an RGB image to greyscale

    Args:
        im ([np.array]): [np.array of shape [H, W, 3]]

    Returns:
        im ([np.array]): [np.array of shape [H, W]]
    """
    # Getting the dimensions of the im matrix
    im_size = im.shape

    im_rows = im_size[0]
    im_cols = im_size[1]

    # Create a greyscale output matrix of size [H, W], initialize with zeros
    greyscale = np.zeros((im_rows, im_cols))

    # Two for-loops traversing the colour input matrix, converting RGB image to greyscale
    for i in range(im_rows):
        for j in range(im_cols):
            greyscale[i,j] = 0.212 * im[i,j][0] + 0.7152 * im[i,j][1] + 0.0722 * im[i,j][2]

    im = greyscale

    return im


def inverse(im):
    """
    Finds the inverse of the greyscale image

    Args:
<<<<<<< HEAD
        im ([type]): [np.array of shape [H, W]]

    Returns:
        im ([type]): [np.array of shape [H, W]]
    """
    # Getting the dimensions of the im matrix
    im_size = im.shape

    im_rows = im_size[0]
    im_cols = im_size[1]

    # Create a inverse output matrix of size [H, W], initialize with zeros
    inverse_im = np.zeros((im_rows, im_cols))

    # Two for-loops traversing the input matrix, converting to inverse image
    for i in range(im_rows):
        for j in range(im_cols):
            inverse_im[i,j] = (255 - im[i,j])

    im = inverse_im

=======
        im ([np.array]): [np.array of shape [H, W]]

    Returns:
        im ([np.array]): [np.array of shape [H, W]]
    """
     # YOUR CODE HERE
>>>>>>> upstream/master
    return im


if __name__ == "__main__":
    im = plt.imread("images/lake.jpg")

    im = greyscale(im)
    inverse_im = inverse(im)

    save_im("lake_greyscale.jpg", im, cmap="gray")
    save_im("lake_inverse.jpg", inverse_im, cmap="gray")
