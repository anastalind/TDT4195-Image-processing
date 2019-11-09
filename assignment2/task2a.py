import skimage
import numpy as np
import utils


def MaxPool2d(im: np.array,
              kernel_size: int,
    ):
    """ A function that max pools an image with size kernel size.
    Assume that the stride is equal to the kernel size, and that the kernel size is even.

    Args:
        im: [np.array of shape [H, W, 3]]
        kernel_size: integer
    Returns:
        im: [np.array of shape [H/kernel_size, W/kernel_size, 3]].
    """
    stride = kernel_size
    ### START YOUR CODE HERE ### (You can change anything inside this block)
    # Getting the dimensions of the im matrix
    im_height = im.shape[0]
    im_width = im.shape[1]
    im_depth = im.shape[2]

    # Setting dimensions of new im matrix
    new_im_height = int(im_height/kernel_size)
    new_im_width = int(im_width/kernel_size)
    new_im_depth = im_depth

    # Creating output matrix
    new_im = np.zeros((new_im_height, new_im_width, new_im_depth))

    print("Maxpoolin' \n")

    for i in range(new_im_height):
        for j in range(new_im_width):
            for k in range(new_im_depth):
                new_im[i, j, k] = np.max(im[(i * stride):(i * stride + kernel_size), (j * stride):(j * stride + kernel_size), k])

    print("Done Maxpoolin' \n")

    return new_im
    ### END YOUR CODE HERE ###


if __name__ == "__main__":

    # DO NOT CHANGE
    im = skimage.data.chelsea()
    im = utils.uint8_to_float(im)
    max_pooled_image = MaxPool2d(im, 4)

    utils.save_im("chelsea.png", im)
    utils.save_im("chelsea_maxpooled.png", max_pooled_image)

    im = utils.create_checkerboard()
    im = utils.uint8_to_float(im)
    utils.save_im("checkerboard.png", im)
    max_pooled_image = MaxPool2d(im, 2)
    utils.save_im("checkerboard_maxpooled.png", max_pooled_image)
