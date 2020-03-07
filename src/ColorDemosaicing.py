import numpy as np
import os
import matplotlib.pyplot as plt

#
# this function will load in the raw mosaiced data stored in the pgm file
#


def read_pgm(filename):
    """
    Return image data from a raw PGM file as a numpy array
    Format specification: http://netpbm.sourceforge.net/doc/pgm.html

    """
    infile = open(filename, 'r', encoding="ISO-8859-1")

    # read in header
    magic = infile.readline()
    width, height = [int(item) for item in infile.readline().split()]
    maxval = infile.readline()

    # read in image data and reshape to 2D array, convert 16bit to float
    image = np.fromfile(infile, dtype='>u2').reshape((height, width))
    image = image.astype(float) / 65535.
    return image


def demosaic(I):
    """
    Demosaic a Bayer RG/GB image to an RGB image.

    Parameters
    ----------
    I : numpy.array  (dtype=float)
        RG/GB mosaic image of size  HxW

    Returns
    -------
    numpy.array (dtype=float)
    HxWx3 array containing the demosaiced RGB image

    """

    ##############
    # RED
    ##############
    # Initialize array with zeros to start
    my_red = np.zeros((I.shape[0], I.shape[1]))

    # Copies the pixels which have actual red values stored
    my_red[::2, ::2] = I[::2, ::2]

    # Left and Right Interpolation
    my_red[::2, 1:-1:2] = (my_red[::2, 0:-3:2] + my_red[::2, 2:-1:2]) / 2

    # Above and Below Interpolation
    my_red[1:-1:2, ::2] = (my_red[0:-3:2, ::2] + my_red[2:-1:2, ::2]) / 2

    # Bottom Edge Interpolation
    my_red[-1, :] = my_red[-2, :]

    # Right Edge Interpolation
    my_red[:, -1] = my_red[:, -2]

    # Center Interpolation
    my_red[1:-1:2, 1:-2:2] = (my_red[0:-3:2, 1:-2:2] + my_red[2:-1:2, 1:-2:2] + my_red[1:-2:2, 0:-3:2] + my_red[1:-2:2,
                                                                                                                2:-1:2]) / 4

    # Bottom Right Corner Interpolation
    my_red[-1, -1] = (my_red[-1, -2] + my_red[-2, -1]) / 2

    ##############
    # GREEN
    ##############
    # Fill with zeros
    my_green = np.zeros((I.shape[0], I.shape[1]))

    # Copies the pixels which have actual green values stored
    my_green[0::2, 1::2] = I[0::2, 1::2]
    my_green[1::2, 0::2] = I[1::2, 0::2]

    # Left Edge Interpolation
    my_green[2:-1:2, 0] = (my_green[1:-2:2, 0] +
                           my_green[3::2, 0] + my_green[2:-1:2, 1]) / 3

    # Right Edge Interpolation
    my_green[1:-2:2, -1] = (my_green[0:-2:2, -1] +
                            my_green[2::2, -1] + my_green[1:-2:2, -2]) / 3

    # Top Edge Interpolation
    my_green[0, 2:-1:2] = (my_green[0, 1:-2:2] +
                           my_green[0, 3::2] + my_green[1, 2:-1:2]) / 3

    # Bottom Edge Interpolation
    my_green[-1, 1:-2:2] = (my_green[-1, 0:-3:2] +
                            my_green[-1, 2:-1:2] + my_green[-2, 1:-2:2]) / 3

    # Center Interpolation
    my_green[1:-1:2, 1:-1:2] = (my_green[0:-2:2, 1:-1:2] + my_green[2::2, 1:-1:2] + my_green[1:-1:2, 0:-2:2] + my_green[
        1:-1:2,
        2::2]) / 4

    # Center Interpolation
    my_green[2:-1:2, 2:-1:2] = (my_green[1:-2:2, 2:-1:2] + my_green[3::2, 2:-1:2] + my_green[2:-1:2, 1:-2:2] + my_green[
        2:-1:2,
        3::2]) / 4

    # Top Left Corner Interpolation
    my_green[0, 0] = (my_green[0, 1] + my_green[1, 0]) / 2

    # Bottom Right Corner Interpolation
    my_green[-1, -1] = (my_green[-1, -2] + my_green[-2, -1]) / 2

    ###############
    # BLUE
    ###############

    # Initialize array with zeros to start
    my_blue = np.zeros((I.shape[0], I.shape[1]))

    # Copies the pixels which have actual blue values stored
    my_blue[1::2, 1::2] = I[1::2, 1::2]

    # Above and Below Interpolation
    my_blue[2:-1:2, 1::2] = (my_blue[1:-2:2, 1::2] + my_blue[3::2, 1::2]) / 2

    # Left and Right Interpolation
    my_blue[1::2, 2:-1:2] = (my_blue[1::2, 1:-1:2] + my_blue[1::2, 3::2]) / 2

    # Center Interpolation
    my_blue[2:-1:2, 2:-1:2] = (my_blue[2:-1:2, 1:-2:2] + my_blue[2:-1:2, 3::2] + my_blue[1:-2:2, 2:-1:2] + my_blue[3::2,
                                                                                                                   2:-1:2]) / 4

    # Top Edge Interpolation
    my_blue[0, 1:] = my_blue[1, 1:]

    # Right Edge Interpolation
    my_blue[1:, 0] = my_blue[1:, 1]

    # Top Right Corner Interpolation
    my_blue[0, 0] = (my_blue[0, 1] + my_blue[1, 0]) / 2

    # if (I.dtype == np.uint8):
    #     I = I.astype(float) / 256
    # RGB_Array = zip((my_red, my_green, my_blue))

    # Create an empty Image to fill with our 2D arrays
    I = np.zeros((I.shape[0], I.shape[1], 3), dtype=np.float64)
    I[:, :, 0] = my_red
    I[:, :, 1] = my_green
    I[:, :, 2] = my_blue

    return I


# Testing the functions
Iraw = read_pgm("IMG_1308.pgm")
plt.imshow(Iraw, cmap=plt.cm.gray)
plt.show()

Demosaiced = demosaic(Iraw)
plt.imshow(Demosaiced)
plt.show()

JPGFinished = plt.imread('IMG_1308.JPG')
plt.imshow(JPGFinished)
plt.show()
