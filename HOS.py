import cv2
import numpy as np
# from skimage.morphology import skeletonize
s_size=32
def neighbours(x, y, image):
    """Return 8-neighbours of point p1 of picture, in clockwise order"""
    i = image
    x1, y1, x_1, y_1 = x+1, y-1, x-1, y+1
    return [i[y1][x],  i[y1][x1],   i[y][x1],  i[y_1][x1],  # P2,P3,P4,P5
            i[y_1][x], i[y_1][x_1], i[y][x_1], i[y1][x_1]]  # P6,P7,P8,P9


def neighbours_prop(x, y, image):
    """Return 8-neighbours of point p1 of picture, in clockwise order"""
    i = image
    x1 = int(np.std([x + 3, x + 2]))
    y1 = int(np.std([y - 3, y - 2]))
    x_1 = int(np.std([x - 3, x - 2]))
    y_1 = int(np.std([y + 3, y + 2]))
    # x1, y1, x_1, y_1 = x+2, y-2, x-2, y+2
    return [i[y1][x],  i[y1][x1],   i[y][x1],  i[y_1][x1],  # P2,P3,P4,P5
            i[y_1][x], i[y_1][x_1], i[y][x_1], i[y1][x_1]]  # P6,P7,P8,P9


def transitions(neighbours):
    n = neighbours + neighbours[0:1]    # P2, ... P9, P2
    return sum((n1, n2) == (0, 1) for n1, n2 in zip(n, n[1:]))


def skeleton(image, prop=False):
    retval, orig_thresh = cv2.threshold(image, 100, 255, cv2.THRESH_BINARY_INV)
    image = (orig_thresh == 0).astype(np.uint8)
    if prop:
        changing1 = changing2 = [(-1, -1)]
        while changing1 or changing2:
            # Step 1
            changing1 = []
            for y in range(2, len(image) - 2):
                for x in range(2, len(image[0]) - 2):
                    P2, P3, P4, P5, P6, P7, P8, P9 = n = neighbours(x, y, image)
                    if (image[y][x] == 1 and    # (Condition 0)
                        P4 * P6 * P8 == 0 and   # Condition 4
                        P2 * P4 * P6 == 0 and   # Condition 3
                        transitions(n) == 1 and  # Condition 2
                        2 <= sum(n) <= 6):      # Condition 1
                        changing1.append((x, y))
            for x, y in changing1:
                image[y][x] = 0
            # Step 2
            changing2 = []
            for y in range(2, len(image) - 2):
                for x in range(2, len(image[0]) - 2):
                    P2,P3,P4,P5,P6,P7,P8,P9 = n = neighbours_prop(x, y, image)
                    if (image[y][x] == 1 and    # (Condition 0)
                        P2 * P6 * P8 == 0 and   # Condition 4
                        P2 * P4 * P8 == 0 and   # Condition 3
                        transitions(n) == 1 and  # Condition 2
                        2 <= sum(n) <= 6):      # Condition 1
                        changing2.append((x, y))
            for x, y in changing2:
                image[y][x] = 0
        return image * 255
    else:
        changing1 = changing2 = [(-1, -1)]
        while changing1 or changing2:
            # Step 1
            changing1 = []
            for y in range(1, len(image) - 1):
                for x in range(1, len(image[0]) - 1):
                    P2, P3, P4, P5, P6, P7, P8, P9 = n = neighbours(x, y, image)
                    if (image[y][x] == 1 and  # (Condition 0)
                            P4 * P6 * P8 == 0 and  # Condition 4
                            P2 * P4 * P6 == 0 and  # Condition 3
                            transitions(n) == 1 and  # Condition 2
                            2 <= sum(n) <= 6):  # Condition 1
                        changing1.append((x, y))
            for x, y in changing1:
                image[y][x] = 0
            # Step 2
            changing2 = []
            for y in range(1, len(image) - 1):
                for x in range(1, len(image[0]) - 1):
                    P2, P3, P4, P5, P6, P7, P8, P9 = n = neighbours(x, y, image)
                    if (image[y][x] == 1 and  # (Condition 0)
                            P2 * P6 * P8 == 0 and  # Condition 4
                            P2 * P4 * P8 == 0 and  # Condition 3
                            transitions(n) == 1 and  # Condition 2
                            2 <= sum(n) <= 6):  # Condition 1
                        changing2.append((x, y))
            for x, y in changing2:
                image[y][x] = 0
        return image * 255





from skimage.morphology import skeletonize
import matplotlib.pyplot as plt

def hierarchy_skeletonize_image(image):
    # Convert the image to grayscale
    # gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    try:gray=cv2.cvtColor(image,cv2.COLOR_BGR2GRAY)
    except:   gray=image
    # Threshold the image to obtain a binary image
    _, binary_image = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY)
    binary_image[binary_image>1]=1
    # Skeletonize the binary image
    skeleton = skeletonize(binary_image).astype('int')
    nn = np.histogram(skeleton, bins=50)[0]
    return nn


def construct_graph(skeleton):
    import networkx as nx

    # Convert the skeleton image to a graph representation
    graph = nx.Graph()

    # Iterate over each pixel in the skeleton image
    rows, cols = skeleton.shape
    for y in range(rows):
        for x in range(cols):
            # If the pixel is part of the skeleton, add it as a node
            if skeleton[y, x]:
                graph.add_node((x, y))

                # Check neighboring pixels and add edges
                for dy in [-1, 0, 1]:
                    for dx in [-1, 0, 1]:
                        nx, ny = x + dx, y + dy
                        # If the neighboring pixel is also part of the skeleton, add an edge
                        if 0 <= nx < cols and 0 <= ny < rows and skeleton[ny, nx]:
                            graph.add_edge((x, y), (nx, ny))

    return graph


def visualize_graph(graph):
    import networkx as nx
    # Visualize the graph
    pos = nx.spring_layout(graph)
    nx.draw(graph, pos, with_labels=True)
    plt.show()


def hierarchyofskeleton(val,i):
    print('Hos   ',i)
    try:val=cv2.cvtColor(val,cv2.COLOR_BGR2GRAY)
    except:val=val
    data1 = np.resize(val, (s_size, s_size))
    conv_sktn_ = skeleton(data1, True).astype('int32')
    nn = np.histogram(conv_sktn_, bins=50)[0]
    return nn
