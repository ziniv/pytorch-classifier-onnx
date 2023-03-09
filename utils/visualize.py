import cv2
import numpy as np

def visualize(images, classes, batch_idx=0):
    
    img = images[batch_idx].numpy()
    img = (np.transpose(img, (1, 2, 0)) * 255.).astype(np.uint8).copy()
    label = classes[batch_idx].numpy()

    cv2.imshow('img', img)
    cv2.waitKey(0)
    # cv2.imwrite('../test('+str(label)+').jpg', img)