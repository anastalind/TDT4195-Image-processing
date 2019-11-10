import matplotlib.pyplot as plt
import numpy as np
import skimage
import utils




def convolve_im(im: np.array,
                kernel: np.array,
                verbose=True):
    """ Convolves the image (im) with the spatial kernel (kernel),
        and returns the resulting image.

        "verbose" can be used for visualizing different parts of the
        convolution.

        Note: kernel can be of different shape than im.

    Args:
        im: np.array of shape [H, W]
        kernel: np.array of shape [K, K]
        verbose: bool
    Returns:
        im: np.array of shape [H, W]
    """
    ### START YOUR CODE HERE ### (You can change anything inside this block)

    im_height = im.shape[0]
    im_width = im.shape[1]

    kernel_height = kernel.shape[0]
    kernel_width = kernel.shape[1]

    # Pad kernel so it is of size [H, W]
    kernel = np.pad(kernel, ((0, im_height - kernel_height), (0, im_width - kernel_width)), 'constant')

    # Compute the Fourier transform of the image
    fft_im = np.fft.fft2(im)

    # Compute the Fourier transform of the padded kernel
    fft_kernel = np.fft.fft2(kernel)

    # Compute the inverse Fourier transform
    inverse_fft_im = np.fft.ifft2(fft_im * fft_kernel)

    # Returns real values of complex values
    conv_result = np.real(inverse_fft_im)

    if verbose:
        # Use plt.subplot to place two or more images beside eachother
        plt.figure(figsize=(20, 5))

        # plt.subplot(num_rows, num_cols, position (1-indexed))
        plt.subplot(1, 4, 1)
        plt.title("Original image")
        plt.imshow(im, cmap="gray")

        plt.subplot(1, 4, 2)
        plt.title("Absolute value of F(f)")
        plt.imshow(np.fft.fftshift(np.log(np.abs(fft_im))), cmap="gray")

        plt.subplot(1, 4, 3)
        plt.title("Absolute value of F(f*g)")
        plt.imshow(np.fft.fftshift(np.log(np.abs(fft_im * fft_kernel))), cmap="gray")

        plt.subplot(1, 4, 4)
        plt.title("Filtered image")
        plt.imshow(conv_result, cmap="gray")

    ### END YOUR CODE HERE ###
    return conv_result


if __name__ == "__main__":
    verbose = True  # change if you want

    # Changing this code should not be needed
    im = skimage.data.camera()
    im = utils.uint8_to_float(im)

    # DO NOT CHANGE
    gaussian_kernel = np.array([
        [1, 4, 6, 4, 1],
        [4, 16, 24, 16, 4],
        [6, 24, 36, 24, 6],
        [4, 16, 24, 16, 4],
        [1, 4, 6, 4, 1],
    ]) / 256
    image_gaussian = convolve_im(im, gaussian_kernel, verbose)

    # DO NOT CHANGE
    sobel_horizontal = np.array([
        [-1, 0, 1],
        [-2, 0, 2],
        [-1, 0, 1]
    ])
    image_sobelx = convolve_im(im, sobel_horizontal, verbose)

    if verbose:
        plt.show()

    utils.save_im("camera_gaussian.png", image_gaussian)
    utils.save_im("camera_sobelx.png", image_sobelx)
