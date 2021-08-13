import os
import cv2
import numpy as np

def extract_edges(img):

    # Blurring 
    blurred = cv2.bilateralFilter(img,15,150,150)

    # Edge Detection
    v = np.median(blurred)
    sigma = 0.33

    lower = int(max(0,(1-sigma)*v))
    upper = int(min(255,(1+sigma)*v))

    img = np.uint8(img)

    edged = cv2.Canny(img,lower,upper)

    return edged

def make_cut(img, edged):

    index = 0
    for i in range(edged.shape[0]):
        aux = np.sum(edged[i])
        if aux != 0:
            index = i
            break

    if index != 0:
        img = img[index-1:]
        edged = edged[index-1:]
    
    return img, edged

def center_image(img):

    edged = extract_edges(img)

    # Superior cut
    img, edged = make_cut(img, edged)

    edged_trans = edged.transpose()
    img_trans = img.transpose()

    # Left cut
    img_trans, edged_trans = make_cut(img_trans, edged_trans)

    edged_trans_flip = np.flip(edged_trans)
    img_trans_flip = np.flip(img_trans)

    # Right cut
    img_trans_flip, edged_trans_flip = make_cut(img_trans_flip, edged_trans_flip)

    edged_trans_flip_trans = edged_trans_flip.transpose()
    img_trans_flip_trans = img_trans_flip.transpose()

    # Inferior cut
    img_trans_flip_trans, edged_trans_flip_trans = make_cut(img_trans_flip_trans,
                                                            edged_trans_flip_trans)
    
    img = np.flip(img_trans_flip_trans.transpose()).transpose()
    edged = np.flip(edged_trans_flip_trans.transpose()).transpose()

    return img, edged

def preprocessing(img, resize, blur, grayscale, rescale, edges, center):
    
    # Convert it to GrayScale or to RGB
    if grayscale:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        # cv2.imwrite('metal_grayscale.jpg',img)

        # Recale Values between 0-1
        if rescale and not edges:
            img_res = img/255
            # cv2.imwrite('metal_rescaled.jpg', img_res)

    else:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    if center:
        img, edged = center_image(img)
        # cv2.imwrite('metal_centered.jpg', img)
        # cv2.imwrite('metal_edges.jpg', edged)
        if edges:
            img = edged

    # Edges and blur
    if edges and not center:
        img = extract_edges(img)
    elif blur:
        img = cv2.bilateralFilter(img,15,50,150)
        # cv2.imwrite('metal_blurred.jpg', img)

    # Resize it to certain dimensions
    if resize[0] != 0 and resize[1] != 1:
        img = cv2.resize(img, resize, interpolation = cv2.INTER_AREA)
        # cv2.imwrite('metal_resized.jpg', img)

    return img

def load_images(root = '~/data/', resize = (0,0), blur = False,
    grayscale = False, rescale = False, edges = False, center = False):

    images = {}

    # Go through categories in the folders
    for label in os.listdir(root):
        label_path = os.path.join(root, label)
        category = []

        # Go through each image in the folder
        for image in os.listdir(label_path):
            # Read Image
            img = cv2.imread(os.path.join(label_path, image))
            img = np.array(img, np.float32)

            img = preprocessing(img, resize, blur, grayscale, rescale,
                                edges, center)

            if img is not None:
                category.append(img)

        images[label] = category

    return images

def main():
    print('Before')
    cwd = os.getcwd()
    root_folder = os.path.dirname(os.path.dirname(os.path.dirname(cwd)))
    data_folder = os.path.join(root_folder,'data')

    images = load_images(root =data_folder)
    print('Done')

main()