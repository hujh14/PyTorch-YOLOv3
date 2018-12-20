import os
import random
import numpy as np
from PIL import Image

import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.ticker import NullLocator

def visualize_image(img, detections):
    pass

def rescale_bboxes(img, detections):
    pass

def visualize(imgs, img_detections, classes, img_size):
    # Bounding-box colors
    cmap = plt.get_cmap('tab20b')
    colors = [cmap(i) for i in np.linspace(0, 1, 20)]

    print ('\nSaving images:')
    # Iterate through images and save plot of detections
    for img_i, (path, detections) in enumerate(zip(imgs, img_detections)):

        print ("(%d) Image: '%s'" % (img_i, path))

        # Create plot
        img = np.array(Image.open(path))
        plt.figure()
        fig, ax = plt.subplots(1)
        ax.imshow(img)

        # The amount of padding that was added
        pad_x = max(img.shape[0] - img.shape[1], 0) * (img_size / max(img.shape))
        pad_y = max(img.shape[1] - img.shape[0], 0) * (img_size / max(img.shape))
        # Image height and width after padding is removed
        unpad_h = img_size - pad_y
        unpad_w = img_size - pad_x

        # Draw bounding boxes and labels of detections
        if detections is not None:
            unique_labels = detections[:, -1].cpu().unique()
            n_cls_preds = len(unique_labels)
            bbox_colors = random.sample(colors, n_cls_preds)
            for x1, y1, x2, y2, conf, cls_conf, cls_pred in detections:

                print ('\t+ Label: %s, Conf: %.5f' % (classes[int(cls_pred)], cls_conf.item()))

                # Rescale coordinates to original dimensions
                box_h = ((y2 - y1) / unpad_h) * img.shape[0]
                box_w = ((x2 - x1) / unpad_w) * img.shape[1]
                y1 = ((y1 - pad_y // 2) / unpad_h) * img.shape[0]
                x1 = ((x1 - pad_x // 2) / unpad_w) * img.shape[1]

                color = bbox_colors[int(np.where(unique_labels == int(cls_pred))[0])]
                # Create a Rectangle patch
                bbox = patches.Rectangle((x1, y1), box_w, box_h, linewidth=2,
                                        edgecolor=color,
                                        facecolor='none')
                # Add the bbox to the plot
                ax.add_patch(bbox)
                # Add label
                plt.text(x1, y1, s=classes[int(cls_pred)], color='white', verticalalignment='top',
                        bbox={'color': color, 'pad': 0})

        # Save generated image with detections
        plt.axis('off')
        plt.gca().xaxis.set_major_locator(NullLocator())
        plt.gca().yaxis.set_major_locator(NullLocator())
        plt.savefig('output/%d.png' % (img_i), bbox_inches='tight', pad_inches=0.0)
        plt.close()
