import numpy as np
import os
import math as m
import matplotlib.pyplot as plt
from task2ab import save_im



def convolve_im(im, kernel):
    """ A function that convolves im with kernel

    Args:
        im ([type]): [np.array of shape [H, W, 3]]
        kernel ([type]): [np.array of shape [K, K]]

    Returns:
        [type]: [np.array of shape [H, W, 3]. should be same as im]
    """
    # Getting the dimensions of the im matrix
    im_size = im.shape

    im_rows = im_size[0]
    im_cols = im_size[1]

    # Creating output matrix
    convolved_im = np.zeros((im_rows, im_cols, 3))

    # Flip kernel
    kernel = np.flipud(np.fliplr(kernel))

    # Find the center of the kernel
    kernel_size = kernel.shape

    kernel_rows = kernel_size[0]
    kernel_cols = kernel_size[1]

    kernel_center = (m.floor(kernel_rows/2), m.floor(kernel_cols/2))

    # Pad the image with the appropriate amount of rows and cols
    im_padded = np.zeros((im_rows + (kernel_rows-1), im_cols + (kernel_cols-1), 3))
    im_padded[kernel_center[0]:-kernel_center[0], kernel_center[1]:-kernel_center[1]]

    for row in range(im_rows):
        for col in range(im_cols):
            # Slicing im_padded matrix to extract a sub matrix of same size as kernel with first element in (row, col)
            sub_matrix_R = im_padded[row:(row + kernel_rows), col:(col + (kernel_cols)), 0]
            sub_matrix_G = im_padded[row:(row + kernel_rows), col:(col + (kernel_cols)), 1]
            sub_matrix_B = im_padded[row:(row + kernel_rows), col:(col + (kernel_cols)), 2]

            # Multiply each element of submatrix with kernel, sum the elements and store in image_padded
            convolved_im[row, col, 0] = (sub_matrix_R * kernel).sum()
            convolved_im[row, col, 1] = (sub_matrix_G * kernel).sum()
            convolved_im[row, col, 2] = (sub_matrix_B * kernel).sum()


    return convolved_im


if __name__ == "__main__":
    # Read image
    impath = os.path.join("images", "lake.jpg")
    im = plt.imread(impath)

    # Define the convolutional kernels
    h_a = np.ones((3, 3)) / 9
    h_b = np.array([
        [1, 4, 6, 4, 1],
        [4, 16, 24, 16, 4],
        [6, 24, 36, 24, 6],
        [4, 16, 24, 16, 4],
        [1, 4, 6, 4, 1],
    ]) / 256
    # Convolve images
    smoothed_im1 = convolve_im(im.copy(), h_a)
    smoothed_im2 = convolve_im(im, h_b)

    # DO NOT CHANGE
    assert isinstance(smoothed_im1, np.ndarray), \
        f"Your convolve function has to return a np.array. " +\
        f"Was: {type(smoothed_im1)}"
    assert smoothed_im1.shape == im.shape, \
        f"Expected smoothed im ({smoothed_im1.shape}" + \
        f"to have same shape as im ({im.shape})"
    assert smoothed_im2.shape == im.shape, \
        f"Expected smoothed im ({smoothed_im1.shape}" + \
        f"to have same shape as im ({im.shape})"

    save_im("convolved_im_h_a.jpg", smoothed_im1)
    save_im("convolved_im_h_b.jpg", smoothed_im2)
