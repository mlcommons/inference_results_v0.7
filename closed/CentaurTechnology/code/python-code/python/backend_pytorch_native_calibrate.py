"""
pytoch native backend for post-training quantization calibration
"""
# pylint: disable=unused-argument,missing-docstring
import torch  # currently supports pytorch1.0
import backend

import os

class BackendPytorchNativeCalibrate(backend.Backend):
    def __init__(self):
        super(BackendPytorchNativeCalibrate, self).__init__()
        self.sess = None
        self.model = None
        self.device = "cpu"
        self.warmup_samples = 4
        self.calibration_samples = 500
        self.current_sample_count = 0

    def version(self):
        return torch.__version__

    def name(self):
        return "pytorch-native-calibrate"

    def image_format(self):
        return "NCHW"

    def load(self, model_path, inputs=None, outputs=None):
        self.model = torch.load(model_path,map_location=lambda storage, loc: storage)
        self.model.fuse_model()
        self.model.eval()
        print(self.model)
        # find inputs from the model if not passed in by config
        if inputs:
            self.inputs = inputs
        else:
            self.inputs = []
            initializers = set()
            for i in self.model.graph.initializer:
                initializers.add(i.name)
            for i in self.model.graph.input:
                if i.name not in initializers:
                    self.inputs.append(i.name)
        # find outputs from the model if not passed in by config
        if outputs:
            self.outputs = outputs
        else:
            self.outputs = []
            for i in self.model.graph.output:
                self.outputs.append(i.name)

        # prepare the backend
        self.model = self.model.to(self.device)

        return self

    def quantize(self):
        # Quantize the model.
        torch.quantization.convert(self.model, inplace=True)

        # torch.save() is not currently supported for quantized modules. See https://github.com/pytorch/pytorch/issues/24045. Please use state_dict or torch.jit serialization.
        #torch.save(self.model, 'resnet34-ssd1200.updated.quantized.pytorch', _use_new_zipfile_serialization=True)

        # Trace the model.
        sample = torch.zeros(1, 3, 1200, 1200)
        traced_module = torch.jit.trace(self.model, sample)
        traced_results = traced_module(sample)
        output_model_path = 'resnet34-ssd1200.updated.quantized.traced.pt'
        torch.jit.save(traced_module, output_model_path)
        #traced_module.save(output_model_path)
        print('Quantized model saved to: ' + os.path.join(os.getcwd(), output_model_path))

        #self.model = torch.jit.load(output_model_path) 
        
    def predict(self, feed):
        print('current_sample_count='+str(self.current_sample_count))
        self.current_sample_count += 1

        key=[key for key in feed.keys()][0]    
        feed[key] = torch.tensor(feed[key]).float().to(self.device)
        with torch.no_grad():
            output = self.model(feed[key])    

        if self.current_sample_count == self.warmup_samples: # Skip warmup samples.
            print('Inserting observers into the model for calibration...')
            self.model.qconfig = torch.quantization.get_default_qconfig("fbgemm")
            torch.quantization.prepare(self.model, inplace=True)

        # If have seen all the calibration samples, quantize the model.
        if self.current_sample_count == self.warmup_samples + self.calibration_samples: 
            print('Using the calibration samples that we\'ve seen to quantize the model.')
            self.quantize()

        return output
