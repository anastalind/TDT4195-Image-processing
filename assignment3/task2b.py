import utils
import numpy as np

def find_8_neighbourhood(shape, row, col):
    neighbourhood = []

    max_row = shape[0] - 1
    max_col = shape[1] - 1

    # Top left
    y = min(max(0, row - 1), max_row)
    x = min(max(0, col - 1), max_col)
    neighbourhood.append((y, x))

    # Top center
    y = min(max(0, row - 1), max_row)
    x = col
    neighbourhood.append((y, x))

    # Top right
    y = min(max(0, row - 1), max_row)
    x = min(max(0, col + 1), max_col)
    neighbourhood.append((y, x))

    # Left
    y = row
    x = min(max(0, col - 1), max_col)
    neighbourhood.append((y, x))

    # Right
    y = row
    x = min(max(0, col + 1), max_col)
    neighbourhood.append((y, x))

    # Bottom left
    y = min(max(0, row + 1), max_row)
    x = min(max(0, col - 1), max_col)
    neighbourhood.append((y, x))

    # Bottom center
    y = min(max(0, row + 1), max_row)
    x = col
    neighbourhood.append((y, x))

    # Bottom right
    y = min(max(0, row + 1), max_row)
    x = min(max(0, col + 1), max_col)
    neighbourhood.append((y, x))

    return neighbourhood

def region_growing(im: np.ndarray, seed_points: list, T: int) -> np.ndarray:
    """
        A region growing algorithm that segments an image into 1 or 0 (True or False).
        Finds candidate pixels with a Moore-neighborhood (8-connectedness).
        Uses pixel intensity thresholding with the threshold T as the homogeneity criteria.
        The function takes in a grayscale image and outputs a boolean image

        args:
            im: np.ndarray of shape (H, W) in the range [0, 255] (dtype=np.uint8)
            seed_points: list of list containing seed points (row, col). Ex:
                [[row1, col1], [row2, col2], ...]
            T: integer value defining the threshold to used for the homogeneity criteria.
        return:
            (np.ndarray) of shape (H, W). dtype=np.bool
    """
    ### START YOUR CODE HERE ### (You can change anything inside this block)
    # You can also define other helper functions

    segmented = np.zeros_like(im).astype(bool)


    for seed_row, seed_col in seed_points:
        candidates = [(seed_row, seed_col)]

        while len(candidates) > 0:
            row, col = candidates.pop(0)

            # Converting np.uint8 to ints to avoid overflow error
            a, b = map(int, (im[seed_row, seed_col], im[row, col]))

            # Calculate the absolute difference between the neighbour pixel and the center pixel
            abs_diff = np.abs(a - b)

            # Add to segmented image if not already in it and if the absolute difference is less than the threshold
            if not segmented[row, col] and (abs_diff <= T):
                segmented[row, col] = True
                candidates += find_8_neighbourhood(im.shape, row, col)

    return segmented
    ### END YOUR CODE HERE ###



if __name__ == "__main__":
    # DO NOT CHANGE
    im = utils.read_image("defective-weld.png")

    seed_points = [ # (row, column)
        [254, 138], # Seed point 1
        [253, 296], # Seed point 2
        [233, 436], # Seed point 3
        [232, 417], # Seed point 4
    ]
    intensity_threshold = 50
    segmented_image = region_growing(im, seed_points, intensity_threshold)

    assert im.shape == segmented_image.shape, \
        "Expected image shape ({}) to be same as thresholded image shape ({})".format(
            im.shape, segmented_image.shape)
    assert segmented_image.dtype == np.bool, \
        "Expected thresholded image dtype to be np.bool. Was: {}".format(
            segmented_image.dtype)

    segmented_image = utils.to_uint8(segmented_image)
    utils.save_im("defective-weld-segmented.png", segmented_image)
