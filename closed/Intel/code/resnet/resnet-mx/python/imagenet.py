"""
implementation of imagenet dataset
"""

# pylint: disable=unused-argument,missing-docstring

import logging
import os
import re
import time
import multiprocessing
from multiprocessing.pool import ThreadPool

import cv2
import numpy as np

import dataset

logging.basicConfig(level=logging.INFO)
log = logging.getLogger("imagenet")


class Imagenet(dataset.Dataset):

    def __init__(self, data_path, image_list, name, use_cache=0, image_size=None,
                 image_format="NHWC", pre_process=None, count=None, cache_dir=None,
                 use_int8=False, num_workers=1):
        super(Imagenet, self).__init__()
        if image_size is None:
            self.image_size = [224, 224, 3]
        else:
            self.image_size = image_size
        if not cache_dir:
            cache_dir = os.getcwd()
        self.image_list = []
        self.label_list = []
        self.count = count
        self.use_cache = use_cache
        self.use_int8 = use_int8
        self.cache_dir = os.path.join(cache_dir, "preprocessed", name, image_format)
        self.data_path = data_path
        self.pre_process = pre_process
        # input images are in HWC
        self.need_transpose = True if image_format == "NCHW" else False
        self.num_workers = num_workers
        self.scale_factor = 1.0

        if image_list is None:
            # by default look for val_map.txt
            image_list = os.path.join(data_path, "val_map.txt")

        os.makedirs(self.cache_dir, exist_ok=True)

        log.info("Start to preprocess dataset...")
        with open(image_list, 'r') as f:
            for s in f:
                image_name, label = re.split(r"\s+", s.strip())
                self.image_list.append(image_name)
                self.label_list.append(int(label))

                # limit the dataset if requested
                if self.count and len(self.image_list) >= self.count:
                    break

        if not self.image_list:
            log.error("no images in image list found")
            raise ValueError("no images in image list found")

        self.lock = multiprocessing.Lock()
        self.not_found = multiprocessing.Value("i", 0)
        self.start_preprocess()

    def start_preprocess(self):
        def process_image(image_name):
            src = os.path.join(self.data_path, image_name)
            if not os.path.exists(src):
                # if the image does not exists ignore it
                self.lock.acquire()
                self.not_found.value += 1
                self.lock.release()
                return
            dst = os.path.join(self.cache_dir, image_name)
            if not os.path.exists(dst + ".npy"):
                img_org = cv2.imread(src)
                processed = self.pre_process(img_org, need_transpose=self.need_transpose,
                                             dims=self.image_size)
                np.save(dst, processed)

        start = time.time()
        thread_pool = ThreadPool(self.num_workers)
        thread_pool.map(process_image, self.image_list)
        time_taken = time.time() - start

        if self.not_found.value > 0:
            log.info("reduced image list, %d images not found", self.not_found.value)
        log.info("loaded {} images, cache={}, took={:.1f}sec".format(
            len(self.image_list), self.use_cache, time_taken))

        self.label_list = np.array(self.label_list)

    def quantize_dataset(self, quantize_fn=None):
        def process_image(image_name):
            src = os.path.join(self.data_path, image_name)
            if not os.path.exists(src):
                # if the image does not exists ignore it
                self.lock.acquire()
                self.not_found.value += 1
                self.lock.release()
                return
            dst = os.path.join(self.cache_dir, image_name)
            fp32_arr_file = dst + ".npy"
            if not os.path.exists(fp32_arr_file):
                img_org = cv2.imread(src)
                processed = self.pre_process(img_org, need_transpose=self.need_transpose,
                                             dims=self.image_size)
                np.save(fp32_arr_file, processed)

            int8_arr_file = dst + ".int8.npy"
            if not os.path.exists(int8_arr_file):
                # convert from FP32 numpy array to INT8
                data_fp32 = np.load(fp32_arr_file)
                if quantize_fn is None:
                    data_int8 = np.clip(data_fp32 * self.scale_factor, a_min=-127, a_max=127).astype('int8')
                    np.save(int8_arr_file, data_int8)
                else:
                    data_int8 = quantize_fn(data_fp32)
                    np.save(int8_arr_file, data_int8)

        start = time.time()
        thread_pool = ThreadPool(self.num_workers)
        thread_pool.map(process_image, self.image_list)
        time_taken = time.time() - start

        if self.not_found.value > 0:
            log.info("reduced image list, %d images not found", self.not_found.value)
        log.info("quantize {} images, cache={}, took={:.1f}sec".format(
            len(self.image_list), self.use_cache, time_taken))

    def get_item(self, nr):
        """Get image by number in the list."""
        dst = os.path.join(self.cache_dir, self.image_list[nr])
        if self.use_int8:
            dst += ".int8"
        img = np.load(dst + ".npy")
        return img, self.label_list[nr]

    def get_item_loc(self, nr):
        src = os.path.join(self.data_path, self.image_list[nr])
        return src