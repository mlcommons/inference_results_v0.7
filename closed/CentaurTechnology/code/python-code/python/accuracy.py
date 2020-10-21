"""
implementation of imagenet dataset
"""

# pylint: disable=unused-argument,missing-docstring

import logging
import json
import logging
import os

import cv2
import numpy as np

#import dataset

logging.basicConfig(level=logging.WARNING)

dtype_map = {
    "float32": np.float32,
    "int32": np.int32,
    "int64": np.int64
}


#============================================================
class ImagenetAcc():

    def __init__(self, mlperf_accuracy_file, imagenet_val_file, verbose=False, dtype="float32"):
        self.accuracy_str = ""

        dtype_map = {
            "float32": np.float32,
            "int32": np.int32,
            "int64": np.int64
        }

        imagenet = []
        with open(imagenet_val_file, "r") as f:
            for line in f:
                cols = line.strip().split()
                imagenet.append((cols[0], int(cols[1])))

        with open(mlperf_accuracy_file, "r") as f:
            results = json.load(f)

        seen = set()
        good = 0
        for j in results:
            idx = j['qsl_idx']

            # de-dupe in case loadgen sends the same image multiple times
            if idx in seen:
                continue
            seen.add(idx)

            # get the expected label and image
            img, label = imagenet[idx]

            # reconstruct label from mlperf accuracy log
            data = np.frombuffer(bytes.fromhex(j['data']), dtype_map[dtype])
            found = int(data[0])
            if label == found:
                good += 1
            else:
                if verbose:
                    self.accuracy_str += ("{}, expected: {}, found {}".format(img, label, found))

        self.accuracy_str += ("accuracy={:.3f}%, good={}, total={}".format(100. * good / len(seen), good, len(seen)))
        if verbose:
            self.accuracy_str += ("found and ignored {} dupes".format(len(results) - len(seen)))

    def get_accuracy(self):
        return self.accuracy_str

#============================================================
class CocoAcc():

    def __init__(self, mlperf_accuracy_file, coco_dir, verbose=False, output_file="coco-results.json", use_inv_map=False, remove_48_empty_images=False):
        from pycocotools.coco import COCO
        from pycocotools.cocoeval import COCOeval

        self.accuracy_str = ""
        self.detections = []

        cocoGt = COCO(os.path.join(coco_dir, "annotations/instances_val2017.json"))

        if use_inv_map:
            inv_map = [0] + cocoGt.getCatIds() # First label in inv_map is not used

        with open(mlperf_accuracy_file, "r") as f:
            results = json.load(f)

        image_ids = set()
        seen = set()
        no_results = 0
        if remove_48_empty_images:        
            im_ids = []
            for i in cocoGt.getCatIds():
                im_ids += cocoGt.catToImgs[i]
            im_ids = list(set(im_ids))
            image_map = [cocoGt.imgs[id] for id in im_ids]
        else:
            image_map = cocoGt.dataset["images"]

        for j in results:
            idx = j['qsl_idx']
            # de-dupe in case loadgen sends the same image multiple times
            if idx in seen:
                continue
            seen.add(idx)

            # reconstruct from mlperf accuracy log
            # what is written by the benchmark is an array of float32's:
            # id, box[0], box[1], box[2], box[3], score, detection_class
            # note that id is a index into instances_val2017.json, not the actual image_id
            data = np.frombuffer(bytes.fromhex(j['data']), np.float32)
            if len(data) < 7:
                # handle images that had no results
                image = image_map[idx]
                # by adding the id to image_ids we make pycoco aware of the no-result image
                image_ids.add(image["id"])
                no_results += 1
                if verbose:
                    self.accuracy_str += ("no results: {}, idx={}".format(image["coco_url"], idx))
                continue

            for i in range(0, len(data), 7):
                image_idx, ymin, xmin, ymax, xmax, score, label = data[i:i + 7]
                image = image_map[idx]
                image_idx = int(image_idx)
                if image_idx != idx:
                    self.accuracy_str += ("ERROR: loadgen({}) and payload({}) disagree on image_idx".format(idx, image_idx))
                image_id = image["id"]
                height, width = image["height"], image["width"]
                ymin *= height
                xmin *= width
                ymax *= height
                xmax *= width
                loc = os.path.join(coco_dir, "val2017", image["file_name"])
                label = int(label)
                if use_inv_map:
                    label = inv_map[label]
                # pycoco wants {imageID,x1,y1,w,h,score,class}
                self.detections.append({
                    "image_id": image_id,
                    "image_loc": loc,
                    "category_id": label,
                    "bbox": [float(xmin), float(ymin), float(xmax - xmin), float(ymax - ymin)],
                    "score": float(score)})
                image_ids.add(image_id)

        with open(output_file, "w") as fp:
            json.dump(self.detections, fp, sort_keys=True, indent=4)

        cocoDt = cocoGt.loadRes(output_file) # Load from file to bypass error with Python3
        cocoEval = COCOeval(cocoGt, cocoDt, iouType='bbox')
        cocoEval.params.imgIds = list(image_ids)
        cocoEval.evaluate()
        cocoEval.accumulate()
        cocoEval.summarize()

        self.accuracy_str += ("mAP={:.3f}%".format(100. * cocoEval.stats[0]))
        if verbose:
            self.accuracy_str += ("found {} results".format(len(results)))
            self.accuracy_str += ("found {} images".format(len(image_ids)))
            self.accuracy_str += ("found {} images with no results".format(no_results))
            self.accuracy_str += ("ignored {} dupes".format(len(results) - len(seen)))

    def get_accuracy(self):
        return self.accuracy_str

    def get_detections(self):
        return self.detections

