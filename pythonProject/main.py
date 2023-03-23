import queue

import cv2 as cv

import numpy as np

images_total = 15
ring_iteration = 1


def threshold_otsu_impl(image):
    bins_num = 256

    # Get the image histogram
    hist, bin_edges = np.histogram(image, bins=bins_num)

    # Get normalized histogram if it is required

    hist = np.divide(hist.ravel(), hist.max())

    # Calculate centers of bins
    bin_mids = (bin_edges[:-1] + bin_edges[1:]) / 2.

    # Iterate over all thresholds (indices) and get the probabilities w1(t), w2(t)
    wB = np.cumsum(hist)
    wF = np.cumsum(hist[::-1])[::-1]

    # Get the class means mu0(t)
    mean1 = np.cumsum(hist * bin_mids) / wB
    # Get the class means mu1(t)
    mean2 = (np.cumsum((hist * bin_mids)[::-1]) / wF[::-1])[::-1]

    inter_class_variance = wB[:-1] * wF[1:] * (mean1[:-1] - mean2[1:]) ** 2

    # Maximize the inter_class_variance function val
    index_of_max_val = np.argmax(inter_class_variance)

    threshold = bin_mids[:-1][index_of_max_val]

    return threshold


def thresholdImage(image, threshold):
    for i in range(0, image.shape[0]):
        for j in range(0, image.shape[1]):
            if image[i, j] > threshold:
                image[i, j] = 0
            else:
                image[i, j] = 255

    return image


def dilation(image):
    copy = image.copy()
    S = 1
    for i in range(S, image.shape[0] - S):
        for j in range(S, image.shape[1] - S):
            copy[i, j] = np.max(image[i - S: i + S + 1, j - S: j + S + 1])
    return copy


def erosion(image):
    copy = image.copy()
    # copy = np.full((image.shape[0], image.shape[1]), 0).astype(np.float64)
    S = 1
    for i in range(S, image.shape[0] - S):
        for j in range(S, image.shape[1] - S):
            copy[i, j] = np.min(image[i - S: i + S + 1, j - S: j + S + 1])
    return copy


# one component at a time implementation from wikipedia
def imageCCL(image):
    labeled_image = image.copy()
    labelID = 1
    labelQueue = queue.Queue()

    for i in range(0, image.shape[0]):
        for j in range(0, image.shape[1]):
            labeled_image[i, j] = 0

    for i in range(0, image.shape[0]):
        for j in range(0, image.shape[1]):
            if (labeled_image[i, j] == 0) and (
                    image[i, j] == 0):
                labeled_image[i, j] = labelID
                labelQueue.put([i, j])

                while not labelQueue.empty():
                    pixel = labelQueue.get()

                    # Fetch the 4 neighbours of this pixel.
                    neighbours = [
                        [pixel[0], pixel[1] - 1],
                        [pixel[0], pixel[1] + 1],
                        [pixel[0] - 1, pixel[1]],
                        [pixel[0] + 1, pixel[1]],
                    ]

                    for neighbour in neighbours:

                        if (labeled_image[neighbour[0], neighbour[1]] == 0) and (
                                image[neighbour[0], neighbour[1]] == 0):
                            labeled_image[neighbour[0], [neighbour[1]]] = labelID
                            labelQueue.put(neighbour)

                labelID += 1
    return labeled_image


def displayImage(labeled_image):
    # display the image.
    label_frequency = [0, 0, 0, 0]
    labeled_image = labeled_image.copy()

    for i in range(0, labeled_image.shape[0]):
        for j in range(0, labeled_image.shape[1]):
            if labeled_image[i, j] > 255:
                label_frequency[labeled_image[i, j]] += 1

    mostFrequent = np.argmax(label_frequency)

    for i in range(0, labeled_image.shape[0]):
        for j in range(0, labeled_image.shape[1]):
            if labeled_image[i, j] > 0 and labeled_image[i, j] != mostFrequent:
                labeled_image[i, j] = 220
            elif labeled_image[i, j] == mostFrequent:
                labeled_image[i, j] = 0

    labeled_image = cv.cvtColor(labeled_image, cv.COLOR_GRAY2RGB)

    cv.imshow("cv", labeled_image)

    cv.waitKey(0)
    cv.destroyAllWindows()


while True:

    theImage = cv.imread("./images/Oring" + str(ring_iteration) + ".jpg", 0)

    threshold = threshold_otsu_impl(theImage)
    theImage = thresholdImage(theImage, threshold)

    theImage = dilation(theImage)
    theImage = erosion(theImage)
    labeled_image = imageCCL(theImage)

    displayImage(labeled_image)

    ring_iteration = ring_iteration + 1

    if ring_iteration >= images_total:
        break
