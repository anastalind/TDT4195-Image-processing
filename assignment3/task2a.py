import numpy as np
import skimage
import utils
import pathlib


def otsu_thresholding(im: np.ndarray) -> int:
    """
        Otsu's thresholding algorithm that segments an image into 1 or 0 (True or False)
        The function takes in a grayscale image and outputs a boolean image

        args:
            im: np.ndarray of shape (H, W) in the range [0, 255] (dtype=np.uint8)
        return:
            (int) the computed thresholding value
    """
    assert im.dtype == np.uint8
    ### START YOUR CODE HERE ### (You can change anything inside this block)
    # You can also define other helper functions

    # Number of intensity levels
    L = 256

    # Compute normalized histogram
    p, _ = np.histogram(im, bins=L, density=True)

    # Compute the cumulative sums P_1(k) for k = 0, 1, 2,.., L-1, the cumulative means m_k for k = 0, 1, 2,.., L-1 and the global mean m_g
    P_1 = np.zeros(L)
    m = np.zeros(L)
    m_g = 0

    for k in range(L):
        # Computes the global mean
        m_g += (k * p[k])

        for i in range(k + 1):
            # Computes the cumulative sums
            P_1[k] += p[i]

            # Computes the cumulative means
            m[k] += (i * p[i])

    var = np.zeros(L)
    threshold = 128
    max_var = 0

    # Compute the between-class variance o^2 = (m_g * P_1 - m_k)^2/(P_1 * (1 - P_1)) for k = 0, 1, 2,.., L-1 and find the k value for the max variance
    for k in range(L):
        var[k] = ((m_g * P_1[k] - m[k])**2)/(P_1[k]*(1-P_1[k]))
        if (var[k] > max_var):
            max_var = var[k]
            threshold = k

    return threshold
    ### END YOUR CODE HERE ###


if __name__ == "__main__":
    # DO NOT CHANGE
    impaths_to_segment = [
        pathlib.Path("thumbprint.png"),
        pathlib.Path("polymercell.png")
    ]
    for impath in impaths_to_segment:
        im = utils.read_image(impath)
        threshold = otsu_thresholding(im)
        print("Found optimal threshold:", threshold)

        # Segment the image by threshold
        segmented_image = (im >= threshold)
        assert im.shape == segmented_image.shape, \
            "Expected image shape ({}) to be same as thresholded image shape ({})".format(
                im.shape, segmented_image.shape)
        assert segmented_image.dtype == np.bool, \
            "Expected thresholded image dtype to be np.bool. Was: {}".format(
                segmented_image.dtype)

        segmented_image = utils.to_uint8(segmented_image)

        save_path = "{}-segmented.png".format(impath.stem)
        utils.save_im(save_path, segmented_image)
