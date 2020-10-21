"""
Tool to calculate accuracy for loadgen accuracy output found in mlperf_log_accuracy.json
We assume that loadgen's query index is in the same order as the images in imagenet2012/val_map.txt.
"""

from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import argparse
import json
import os
import numpy as np
import cv2
#import tensorflow as tf
import math


# pylint: disable=missing-docstring

#height, width = (512, 512)
DEBUG_ = False #True
min_resize_value = 512;
max_resize_value = 512;
crop_height = 512;
crop_width = 512;
    
def preprocess_label(label):
    label = label.astype(np.uint8)
    label = resize_to_range(label = label, min_size=min_resize_value, max_size = max_resize_value)
    
    image_height, image_width = label.shape[:2]
    
    target_height = image_height + max([crop_height - image_height, 0])
    target_width = image_width + max([crop_width - image_width, 0])
    offset_height, offset_width = (0,0)
    pad_value = 0
    
    label = pad_to_bounding_box(label, offset_height, offset_width, target_height, target_width, pad_value)
    return label
        
def pad_to_bounding_box(image, offset_height, offset_width, target_height,
                        target_width, pad_value):
    """Pads the given image with the given pad_value.
  Args:
    image: 3-D tensor with shape [height, width, channels]
    offset_height: Number of rows of zeros to add on top.
    offset_width: Number of columns of zeros to add on the left.
    target_height: Height of output image.
    target_width: Width of output image.
    pad_value: Value to pad the image tensor with.
  Returns:
    3-D tensor of shape [target_height, target_width, channels].
  Raises:
    ValueError: If the shape of image is incompatible with the offset_* or
    target_* arguments.
    """

    height, width = image.shape[:2]
    assert target_height >= height
    assert target_width >= width
    
    after_padding_width = target_width - offset_width - width
    after_padding_height = target_height - offset_height - height

    assert after_padding_height >= 0 and after_padding_width >= 0
    if DEBUG_:
        print("Padding: {},{},{},{}".format(offset_height, after_padding_height, offset_width, after_padding_width))
        
    return cv2.copyMakeBorder(image, offset_height, after_padding_height, offset_width, after_padding_width, cv2.BORDER_CONSTANT, value=0)

def resize_to_range(label=None,
                    min_size=None,
                    max_size=None,
                    factor=1,
                    keep_aspect_ratio=True,
                    align_corners=True,
                    label_layout_is_chw=False,
                    scope=None,
                    method=cv2.INTER_NEAREST):

    min_size = float(min_size) #tf.cast(min_size, tf.float32)
    if max_size is not None:
        max_size = float(max_size) #tf.cast(max_size, tf.float32)
        # Modify the max_size to be a multiple of factor plus 1 and make sure the
        # max dimension after resizing is no larger than max_size.
        if factor is not None:
            max_size = (max_size - (max_size - 1) % factor)

    orig_height, orig_width = label.shape[:2] #resolve_shape(image, rank=3)
    orig_height = float(orig_height) #tf.cast(orig_height, tf.float32)
    orig_width = float(orig_width) #tf.cast(orig_width, tf.float32)
    orig_min_size = min([orig_width, orig_height]) #tf.minimum(orig_height, orig_width)

    # Calculate the larger of the possible sizes
    large_scale_factor = min_size / orig_min_size
    large_height = math.floor(orig_height * large_scale_factor) #tf.cast(tf.floor(orig_height * large_scale_factor), tf.int32)
    large_width = math.floor(orig_width * large_scale_factor) #tf.cast(tf.floor(orig_width * large_scale_factor), tf.int32)
    large_size = [large_width, large_height] #tf.stack([large_height, large_width])

    new_size = large_size
    if max_size is not None:
        # Calculate the smaller of the possible sizes, use that if the larger
        # is too big.
        orig_max_size = max([orig_height, orig_width]) #tf.maximum(orig_height, orig_width)
        small_scale_factor = max_size / orig_max_size
        small_height = math.floor(orig_height * small_scale_factor) #tf.cast(tf.floor(orig_height * small_scale_factor), tf.int32)
        small_width = math.floor(orig_width * small_scale_factor) #tf.cast(tf.floor(orig_width * small_scale_factor), tf.int32)
        small_size = [small_width, small_height] #tf.stack([small_height, small_width])
      
        new_size = small_size if max(large_size) > max_size else large_size
    # Ensure that both output sides are multiples of factor plus one.
    if factor is not None:
        new_size[0] += (factor - (new_size[0] - 1) % factor) % factor
        new_size[1] += (factor - (new_size[1] - 1) % factor) % factor
    if not keep_aspect_ratio:
        # If not keep the aspect ratio, we resize everything to max_size, allowing
        # us to do pre-processing without extra padding.
        new_size = [max(new_size), max(new_size)] #[tf.reduce_max(new_size), tf.reduce_max(new_size)]
    
    if DEBUG_:
        print("Resized: {}".format(new_size))
        
    label = cv2.resize(label, tuple(new_size), interpolation = cv2.INTER_NEAREST)
    return label

def get_args():
    """Parse commandline."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--mlperf-accuracy-file", required=True, help="path to mlperf_log_accuracy.json")
    parser.add_argument("--ade20k-dir", required=True, help="path to ade20k directory")
    parser.add_argument("--num-classes", type=int, default=32, help="Number of classes")
    parser.add_argument("--verbose", action="store_true", help="verbose messages")
    args = parser.parse_args()
    return args

def main():
    args = get_args()

    ade20k_val_file = os.path.join(args.ade20k_dir,"images/validation/val.txt")
    val_label_dir = os.path.join(args.ade20k_dir,"annotations/validation")
    
    ade20k = []
    with open(ade20k_val_file, "r") as f:
        for line in f:
            cols = line.strip().split()
            ade20k.append(cols[0])
            
    print("Number of images: {}".format(len(ade20k)))

    with open(args.mlperf_accuracy_file, "r") as f:
        results = json.load(f)

    seen = set()
    
    tps = np.zeros(args.num_classes)
    fps = np.zeros(args.num_classes)
    fns = np.zeros(args.num_classes)
    
    
    for j in results:
        idx = j['qsl_idx']

        # de-dupe in case loadgen sends the same image multiple times
        if idx in seen:
            continue
        seen.add(idx)
        
        
        predLabel = np.frombuffer(bytes.fromhex(j['data']), np.float32)
        if DEBUG_:
            print("==Processing image {} ==".format(idx))
            
        if idx > len(ade20k):
            print(" Skipping processed index ")
            continue
        imgName = ade20k[idx]
        
        
        gtLabelName, _ = imgName.split('.')
        gtLabelPath    = os.path.join(val_label_dir, gtLabelName+'.png')
        gtLabel        = cv2.imread(gtLabelPath)
        gtLabel        = cv2.cvtColor(gtLabel, cv2.COLOR_BGR2RGB)

        if DEBUG_:
            print(" Original Size: {}".format(gtLabel.shape[:2]))

        #gtLabel        = cv2.resize(gtLabel, (height, width), interpolation = cv2.INTER_NEAREST)
        gtLabel         = preprocess_label(gtLabel)    
        
        gtLabel        = np.asarray(gtLabel)        
        
        gtLabel = gtLabel[:,:,0].astype(np.uint8)
        
        predLabel = predLabel.reshape(512,512).astype(np.uint8)
                
        for c in range(1, args.num_classes):
            tp = np.sum((np.multiply(predLabel==c, gtLabel==c)))
            fp = np.sum(gtLabel[(predLabel==c)*(gtLabel > 0)*(gtLabel < args.num_classes)] != c)
            fn = np.sum(predLabel[gtLabel==c] != c)
            
            tps[c-1] = tps[c-1] + tp
            fps[c-1] = fps[c-1] + fp
            fns[c-1] = fns[c-1] + fn
            
    m_iou = 0
    for c in range(1,args.num_classes):
        m_ciou = tps[c-1]/(tps[c-1] + fps[c-1] + fns[c-1])
        
        print("IoU class {}: \t{:.4f}".format(c, m_ciou))
        m_iou += m_ciou
        
    m_iou /= (args.num_classes-1)
    print('mIoU is {:.3f}%'.format( 100*m_iou))
    
if __name__ == "__main__":
    main()