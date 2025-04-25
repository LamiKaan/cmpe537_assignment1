import os
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
from enum import Enum


class Init(Enum):
    MANUAL = 0
    RANDOM = 1


# Image number to be processed for task1
image_id = 3
# Number of quantized colors
k = 32
# Variable to decide initialization mode of color centers (manually or randomly)
initialization = Init.RANDOM
# Limit for the iterations i of the k-means algorithm
max_iterations = 100


def get_path_for_image(image_id):
    # Get directory of image files relative to this file's directory
    image_dir = os.path.join(os.path.dirname(__file__), 'data', 'images', 'task_1')
    # Get image path for the provided image id
    image_path = os.path.join(image_dir, 'image0{}.jpeg'.format(image_id))

    return image_path


def initialize_manually(image, k):
    plt.imshow(image)
    points = plt.ginput(k, timeout=-1, show_clicks=True)

    # Convert to integer so that points represent specific pixels (ginput gives float values)
    color_centers = np.array(points).astype(np.int64)
    # Also swap the places of columns (ginput gives WxH whereas I work with HxW)
    color_centers[:, [0, 1]] = color_centers[:, [1, 0]]

    return color_centers

def initialize_randomly(image_matrix, k):
    # Create a numpy random number generator instance
    rng = np.random.default_rng(seed=537)

    # Generate k amount of random numbers (floats) between 0 and height of the image
    # Then, floor and convert to integer
    color_centers_heights = np.uint(np.floor(rng.uniform(low=0, high=image_matrix.shape[0], size=k)))
    # Generate k amount of random numbers (floats) between 0 and width of the image
    # Then, floor and convert to integer
    color_centers_widths = np.uint(np.floor(rng.uniform(low=0, high=image_matrix.shape[1], size=k)))

    # Stack height and width vectors column-wise into a matrix of shape [k, 2]
    # Each row representing a color center (x, y)
    color_centers = np.stack((color_centers_heights, color_centers_widths), axis=1)

    return color_centers


def quantize(img, k):
    # Convert the image to a numpy array of shape H x W x 3(RGB)
    image_matrix = np.array(img)
    # Create a new broadcast version of the image matrix of shape (H, W, 1, 3)
    broadcast_image_matrix = image_matrix[:, :, np.newaxis, :].astype(dtype=np.int64)

    # Initialize a cluster map with the same shape as the image where each
    # coordinate (pixel) stores the index (0 to k) of the cluster it belongs to
    initial_clusters = np.zeros(image_matrix.shape[0:2], image_matrix.dtype)

    # Initialize color centers
    if initialization == Init.MANUAL:
        initial_color_centers = initialize_manually(img, k)
    else:
        initial_color_centers = initialize_randomly(image_matrix, k)

    # Create variables to hold final results
    final_clusters = None
    final_color_centers = None

    # Start the k-means algorithm
    input_clusters = initial_clusters
    input_color_centers = initial_color_centers.astype(np.int64)
    iterations = 0
    while True:
        iterations += 1

        # Retrieve RGB color vectors of input color centers of shape k x 3
        input_colors = image_matrix[input_color_centers[:, 0], input_color_centers[:, 1], :]
        # Create a new broadcast version of the input color matrix of shape (1, 1, k, 3)
        broadcast_input_colors = input_colors[np.newaxis, np.newaxis, :, :].astype(dtype=np.int64)

        # We need to calculate the distance of each pixel's color, to the colors of every color center.
        # Instead of doing this one pixel at a time with nested for loops, calculate as a whole via
        # numpy vector operations (the reason why we created broadcast versions of the matrices)
        color_distances = np.sum(((broadcast_image_matrix - broadcast_input_colors) ** 2), axis=3)
        # Return the index of the minimum value along the distance axis (assign each pixel the
        # index of the closest color center)
        output_clusters = np.argmin(color_distances, axis=2)

        # Create a variable to store the new color centers based on new clusters
        output_color_centers = None

        for i in range(k):
            # Retrieve pixel coordinates for the current cluster index
            cluster_i_coordinates = np.argwhere(output_clusters == i)

            # Check if there are pixels assigned to the current cluster (at certain iterations
            # of the algorithm, certain clusters might be empty, especially with large k)
            if len(cluster_i_coordinates) > 0:
                # Take the mean of the coordinates, floor and convert to integer
                new_ith_color_center = np.floor(np.mean(cluster_i_coordinates, axis=0)).astype(np.int64)
            else:
                # If the cluster is empty, keep the old color center
                new_ith_color_center = input_color_centers[i]

            if output_color_centers is None:
                output_color_centers = new_ith_color_center
            else:
                output_color_centers = np.vstack((output_color_centers, new_ith_color_center))

        if np.array_equal(output_clusters, input_clusters) or iterations >= max_iterations:
            final_clusters = output_clusters
            final_color_centers = output_color_centers
            # print(iterations)
            break
        else:
            input_clusters = output_clusters
            input_color_centers = output_color_centers

    # Get the final colors based on the final color centers
    final_colors = image_matrix[final_color_centers[:, 0], final_color_centers[:, 1], :]

    # Initialize the quantized image matrix as zeros
    quantized_image_matrix = np.zeros((image_matrix.shape[0], image_matrix.shape[1], 3), dtype=np.uint8)
    # Populate each pixel of the quantized image matrix with the color of the color center that
    # corresponds to the final cluster that the pixel belongs
    quantized_image_matrix = final_colors[final_clusters]

    # Convert the matrix back to an image and return it
    quantized_image = Image.fromarray(quantized_image_matrix.astype(np.uint8))

    return quantized_image


if __name__ == "__main__":
    # Get path of image for given image id
    image_path = get_path_for_image(image_id)
    # Open the image using Pillow
    image = Image.open(image_path)
    # Quantize image
    quantized_image = quantize(image, k)

    quantized_image.show()
