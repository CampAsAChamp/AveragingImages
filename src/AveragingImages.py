import numpy as np
import os
import matplotlib.pyplot as plt


def average_image(dirname):
    """
    Computes the average of all color images in a specified directory and returns the result.

    The function ignores any images that are not color images and ignores any images that are not
    the same size as the first color image you load in

    Parameters
    ----------
    dirname : str
        Directory to search for images

    Returns
    -------
    numpy.array (dtype=float)
        HxWx3 array containing the average of the images found

    """

    Iaverage = 0
    numPics = 0
    height = 0
    width = 0
    initialHeight = None
    initialWidth = None

    for file in os.listdir(dirname):
        filename = os.path.join(dirname, file)
        if os.path.isfile(filename):
            # Read the file in
            I = plt.imread(filename)
            # Checking if image is color
            if len(I.shape) == 3:
                # Convert to float
                if I.dtype == np.uint8:
                    I = I.astype(float) / 256

                if initialHeight == None and initialWidth == None:
                    initialHeight = I.shape[0]
                    initialWidth = I.shape[1]
                    height = I.shape[0]
                    width = I.shape[1]
                    numPics = 1
                    sum = I
                elif I.shape[0] == height and I.shape[1] and initialHeight != None and initialWidth != None:
                    # Read the image from the filename
                    numPics = numPics + 1
                    sum = np.add(sum, I)
                    Iaverage = sum / numPics

    return Iaverage


# Testing the function
averageImgSet1 = average_image("set1")
plt.imshow(averageImgSet1)
plt.show()

exampleImg1 = plt.imread('set1img01.jpg')
plt.imshow(exampleImg1)
plt.show()

averageImgSet2 = average_image("set2")
plt.imshow(averageImgSet2)
plt.show()

exampleImg2 = plt.imread('set2img01.jpg')
plt.imshow(exampleImg2)
plt.show()
