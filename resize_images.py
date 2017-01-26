import os.path as path
import glob
import numpy as np
from skimage.io import imread, imsave
from skimage.transform import resize



def resize_all(dir, (height, width)):
    for im_path in glob.glob(dir):
        new_im_path = path.dirname(im_path) + '/%sx%s/' % (height, width) + path.basename(im_path)
        imsave(new_im_path, resize(imread(im_path), (height, width)))
        print '%s converted to %s' % (im_path, new_im_path)



resize_all('./data/standardized_images/*.jpg', (128, 128))
