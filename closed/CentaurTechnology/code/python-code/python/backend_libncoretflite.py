"""
tflite-ncore backend (adapted from https://github.com/tensorflow/tensorflow/lite)
"""

# pylint: disable=unused-argument,missing-docstring,useless-super-delegation

import os

from threading import Lock

import time

import backend
import ncoretflite


class BackendTfliteNcore(backend.Backend):
    def __init__(self):
        super(BackendTfliteNcore, self).__init__()
        self.sess = None
        self.lock = Lock()
        self.sample_count = 0 # Debug
        self.max_batchsize = 1024 # main.py will set this from runtime arg

        # Don't change these here, change in the child class definitions
        self.threads_to_use = 4
        self.latency_mode = False
        self.infer_single = ncoretflite.infer_resnet
        self.infer_batch = ncoretflite.infer_batch_resnet

    def version(self):
        return "0.1"

    def name(self):
        return "tflite-ncore"

    def image_format(self):
        # tflite is always NHWC
        return "NHWC"

    def load(self, model_path, inputs=None, outputs=None):
        delegate_path = 'ncore_tf_delegate.so'
        self.tflite_tp = ncoretflite.load_model(model_path, self.threads_to_use, self.max_batchsize, delegate_path, self.latency_mode)
        return self

    def destroy(self):
        ncoretflite.destroy(self.tflite_tp)

    def predict(self, feed):
        self.lock.acquire()

        inputs = feed[self.inputs[0]]
        if len(inputs) == 1:
            res = self.infer_single(self.tflite_tp, inputs)
        else:
            res = self.infer_batch(self.tflite_tp, inputs)

        self.lock.release()
        return res


#========================================
# SingleStream (latency)

class BackendTfliteNcoreResnet(BackendTfliteNcore):
    def __init__(self):
        super(BackendTfliteNcoreResnet, self).__init__()
        self.threads_to_use = 1
        self.latency_mode = True
        self.infer_single = ncoretflite.infer_resnet
        self.infer_batch = ncoretflite.infer_batch_resnet


class BackendTfliteNcoreMobileNetV1(BackendTfliteNcore):
    def __init__(self):
        super(BackendTfliteNcoreMobileNetV1, self).__init__()
        self.threads_to_use = 1
        self.latency_mode = True
        self.infer_single = ncoretflite.infer_mobilenetv1
        self.infer_batch = ncoretflite.infer_batch_mobilenetv1


class BackendTfliteNcoreSSD(BackendTfliteNcore):
    def __init__(self):
        super(BackendTfliteNcoreSSD, self).__init__()
        self.threads_to_use = 1
        self.latency_mode = True
        self.infer_single = ncoretflite.infer_ssd
        self.infer_batch = ncoretflite.infer_batch_ssd


#========================================
# Offline (throughput)

class BackendTfliteNcoreResnetOffline(BackendTfliteNcore):
    def __init__(self):
        super(BackendTfliteNcoreResnetOffline, self).__init__()
        self.threads_to_use = 2
        self.latency_mode = False
        self.infer_single = ncoretflite.infer_resnet
        self.infer_batch = ncoretflite.infer_batch_resnet


class BackendTfliteNcoreMobileNetV1Offline(BackendTfliteNcore):
    def __init__(self):
        super(BackendTfliteNcoreMobileNetV1Offline, self).__init__()
        self.threads_to_use = 2
        self.latency_mode = False
        self.infer_single = ncoretflite.infer_mobilenetv1
        self.infer_batch = ncoretflite.infer_batch_mobilenetv1


class BackendTfliteNcoreSSDOffline(BackendTfliteNcore):
    def __init__(self):
        super(BackendTfliteNcoreSSDOffline, self).__init__()
        self.threads_to_use = 4
        self.latency_mode = False
        self.infer_single = ncoretflite.infer_ssd
        self.infer_batch = ncoretflite.infer_batch_ssd

