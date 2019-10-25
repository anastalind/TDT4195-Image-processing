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
    # Creating output matrix
    convolved_im = np.zeros_like(im)

    # Flip kernel
    kernel = np.flipud(np.fliplr(kernel))

    # Getting the dimensions of the im matrix
    im_rows = im.shape[0]
    im_cols = im.shape[1]

    # Getting the dimensions of the kernel matrix
    kernel_rows = kernel.shape[0]
    kernel_cols = kernel.shape[1]

    # Defining center assuming kernel_rows and kernel_cols are odd numbers
    kernel_center_x = m.floor(kernel_cols/2)
    kernel_center_y = m.floor(kernel_rows/2)

    # Pad the image with the appropriate amount of rows and cols
    image_padded = np.zeros((im_rows + (kernel_rows-1), im_cols + (kernel_cols-1), 3))

    image_padded[kernel_center_y:-kernel_center_y, kernel_center_x:-kernel_center_x, 0] = im[:, :, 0]
    image_padded[kernel_center_y:-kernel_center_y, kernel_center_x:-kernel_center_x, 1] = im[:, :, 1]
    image_padded[kernel_center_y:-kernel_center_y, kernel_center_x:-kernel_center_x, 2] = im[:, :, 2]


    # Iterating through all the cells of the image, convolving with the kernel and placing correctly in the convolved image
    for col in range(im_cols):
        for row in range(im_rows):
            convolved_im[row, col, 0] = (kernel * image_padded[row:row+kernel_rows, col:col+kernel_cols, 0]).sum()
            convolved_im[row, col, 1] = (kernel * image_padded[row:row+kernel_rows, col:col+kernel_cols, 1]).sum()
            convolved_im[row, col, 2] = (kernel * image_padded[row:row+kernel_rows, col:col+kernel_cols, 2]).sum()

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
