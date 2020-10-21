"""
Openvino
"""

# pylint: disable=unused-argument,missing-docstring,useless-super-delegation

from threading import Lock
import os
import subprocess
import zipfile
import time
setupvars_path = '/opt/intel/openvino/bin/setupvars.sh'
if not os.path.exists(setupvars_path):
    raise ModuleNotFoundError('Openvino doesn\'t exist in:', setupvars_path)
# CALL THE VARIABLE INITIALIZATION
subprocess.run([setupvars_path], shell=False, check=True)
import numpy as np
import openvino
from openvino.inference_engine import IECore

import backend


class BackendOpenvino(backend.Backend):
    def __init__(self):
        super(BackendOpenvino, self).__init__()
        self.sess = None
        self.model = None
        self.lock = Lock()

        self.sleep_time = 1

    def version(self):
        try:
            return openvino.__version__
        except AttributeError:
            return 0

    def name(self):
        return "openvino"

    def image_format(self):
        return "NCHW"

    def load(self, model_path, inputs=None, outputs=None):
        ie = IECore()
        model_path = self.unzip_checkpoint_into_xml_and_bin(model_path)
        ie_loaded = ie.read_network(model=model_path, weights=model_path.replace('.xml', '.bin'))
        self.model = ie.load_network(ie_loaded, "CPU")
        self.counter = 0
        # find inputs from the model if not passed in by config
        if inputs:
            self.inputs = inputs
        else:
            self.inputs = [next(iter(self.model.input_info))]

        self.expected_batch_size = int(self.model.input_info[self.inputs[0]].input_data.shape[0])

        # find outputs from the model if not passed in by config
        if outputs:
            self.outputs = outputs
        else:
            self.outputs = next(iter(self.model.outputs))

        return self

    @ staticmethod
    def unzip_checkpoint_into_xml_and_bin(checkpoint_file):
        """In case the provided checkpoint is a serialized one - deserialize it and return the .xml path"""
        if checkpoint_file.endswith('.zip'):
            with zipfile.ZipFile(checkpoint_file, 'r') as zipObj:
                zipObj.extractall(os.path.dirname(checkpoint_file))
                name_list = zipObj.namelist()
                for name in name_list:
                    if name.endswith('.xml'):
                        checkpoint_file = os.path.dirname(checkpoint_file) + '/' + name
            # checkpoint_file = checkpoint_file.replace('.zip', '.xml')
        return checkpoint_file

    def predict(self, feed):

        batch_shape = feed[self.inputs[0]].shape
        if batch_shape[0] < self.expected_batch_size:
            pad = self.expected_batch_size - batch_shape[0]
            feed[self.inputs[0]] = np.concatenate([feed[self.inputs[0]], np.zeros(shape=(pad, *batch_shape[1:]))], axis=0)
            # print('PADDING:', pad)
        res = self.model.infer(feed)
        res = res.values()
        res = list(res)[0]
        if batch_shape[0] < self.expected_batch_size:
            res = res[:batch_shape[0]]
            # print('DE-PADDING:', batch_shape[0])
        return [res]


if __name__ == "__main__":
    model = BackendOpenvino()
    model.load("models/deci-model-batch64_v1.zip")
    import numpy as np
    noise = np.zeros([64,3,224,224])
    import time
    for i in range(100):
        _ = model.predict({model.inputs[0]:noise})
    rep = 1000
    tic = time.time()

    for i in range(rep):
        _ = model.predict({model.inputs[0]:noise})
    print(rep*64/(time.time()-tic))
