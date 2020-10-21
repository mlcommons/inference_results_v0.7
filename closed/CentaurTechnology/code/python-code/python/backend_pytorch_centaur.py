"""
pytoch native backend
"""
# pylint: disable=unused-argument,missing-docstring
import torch  # currently supports pytorch1.0
import nctorch
import backend
import os

def nhwc_to_nchw(tensor):
    return tensor.permute(0,3,1,2).contiguous() # NHWC 0123 -> NCHW 0312

def nchw_to_nhwc(tensor):
    return tensor.permute(0,2,3,1).contiguous() # NCHW 0123 -> NHWC 0231

def nchw_to_nhwc_dims(dims):
    return [dims[0], dims[2], dims[3], dims[1]]

def nhwc_to_nchw_dims(dims):
    return [dims[0], dims[3], dims[1], dims[2]]

class BackendPytorchCentaur(backend.Backend):
    def __init__(self):
        super(BackendPytorchCentaur, self).__init__()
        self.sess = None
        self.model = None
        self.device = "cpu"
        self.sample_count = 0

    def version(self):
        return torch.__version__

    def name(self):
        return "pytorch-centaur"

    def image_format(self):
        return "NCHW"

    def load(self, model_path, inputs=None, outputs=None):
        if 'traced' in model_path: # PyTorch save/load broken for quantized, so need different call:
            # From Microsoft/ONNX MLPerf port: https://github.com/BowenBao/inference/tree/master/cloud/single_stage_detector/pytorch
            torch.ops.load_library('/workspace/ncoresw/mlperf/custom_ops.cpython-37m-x86_64-linux-gnu.so')
            self.model = torch.jit.load(model_path)
        else:
            self.model = torch.load(model_path,map_location=lambda storage, loc: storage)
        self.model.eval()
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
        assert self.device.lower() == "cpu", (
            "Centuar PyTorch backed only supports models from the PyTorch CPU device")
        self.model = self.model.to(self.device)

        # Trace the model to make sure it is valid.
        sample = torch.zeros(1, 3, 1200, 1200)
        traced_module = torch.jit.trace(self.model, sample)
        self.model = traced_module

        # For some reason though, the RecursiveScriptModule classes don't seem to be behaving at the moment
        # so I'm just going to hardcode based off the textual mlir values for now.
        out0_scale, out0_zero_point = (0.060359629999999997, 2)  # backbone add_relu output
        out1_scale, out1_zero_point = (0.045420370000000002, 2)  # dangling conv2
        out2_scale, out2_zero_point = (0.048898650000000002, 2)  # dangling conv4
        out3_scale, out3_zero_point = (0.071682650000000001, 2)  # dangling conv6
        out4_scale, out4_zero_point = (0.082455319999999998, 2)  # dangling conv8
        out5_scale, out5_zero_point = (0.072640129999999997, 0)  # dangling conv10

        # Output tensor shapes that the TorchScript interpreter is responsible for allocating
        # We have to pass the sizes as NCHW to TorchScript interpreter
        out_sizes_nhwc = [
            (1, 150, 150, 256), # out0
            (1, 75, 75, 512),   # out1
            (1, 38, 38, 512),   # out2
            (1, 19, 19, 256),   # out3
            (1, 9, 9, 256),     # out4
            (1, 7, 7, 256),     # out5
        ]
        out_sizes = [nhwc_to_nchw_dims(size) for size in out_sizes_nhwc]

        ssd_r34_tensor_infos = [
            {  # out0
                "shape": out_sizes[0],
                "scales": [out0_scale],
                "zero_points": [out0_zero_point],
            },
            {  # out1
                "shape": out_sizes[1],
                "scales": [out1_scale],
                "zero_points": [out1_zero_point],
            },
            {  # out2
                "shape": out_sizes[2],
                "scales": [out2_scale],
                "zero_points": [out2_zero_point],
            },
            {  # out3
                "shape": out_sizes[3],
                "scales": [out3_scale],
                "zero_points": [out3_zero_point],
            },
            {  # out4
                "shape": out_sizes[4],
                "scales": [out4_scale],
                "zero_points": [out4_zero_point],
            },
            {  # out5
                "shape": out_sizes[5],
                "scales": [out5_scale],
                "zero_points": [out5_zero_point],
            },
        ]

        extra_info = {
            "input_shapes": [sample.shape],
            "compiler_target_type": "reference", # reference, emulator, vcl, silicon
            "enable_opt": False,
            "output_tensor_infos": [
                ssd_r34_tensor_infos[0],  # out0
                ssd_r34_tensor_infos[1],  # out1
                ssd_r34_tensor_infos[2],  # out2
                ssd_r34_tensor_infos[3],  # out3
                ssd_r34_tensor_infos[4],  # out4
                ssd_r34_tensor_infos[5],  # out5
            ],
        }

        ssd_r34_backbone_subgraph_params = [
            (114, 332),   # conversion_node_range
            (134, 256),   # execution_node_range
            ["X.1"],      # execution_input_ssa_names
            ["input.1"],  # execution_output_ssa_names
            ["input.1"],  # conversion_output_ssa_names
            extra_info,
        ]

        ssd_r34_backbone_and_dangling_conv_params = [
            (114, 402),  # conversion_node_range
            (134, 286),  # execution_node_range
            ["X.1"],     # execution_input_ssa_names
            ["input.1", "315", "325", "334", "344", "354"],       # execution_output_ssa_names
            ["input.1", "1058", "1072", "1086", "1100", "1114"],  # conversion_output_ssa_names
            extra_info,
        ]

        ## Compile the model for NcoreEngine (with the set verbosity)
        nctorch.logging.set_reportable_log_level(nctorch.logging.Level.Warning)
        mlir_model = nctorch.compile_subgraph(self.model, *ssd_r34_backbone_and_dangling_conv_params)
        print(f"[NCTORCH]: Model compilation complete, here's the new model: {mlir_model.graph}")

        self.model = mlir_model
        with open("/workspace/tmp/ssd_r34.graph.log", "w") as f:
            f.write(str(self.model.graph))

        return self


    def predict(self, feed):
        self.sample_count += 1

        key=[key for key in feed.keys()][0]
        feed[key] = torch.tensor(feed[key]).float().to(self.device)

        ## Return the output tuple directly to MLPerf framework
        return self.model(nchw_to_nhwc(feed[key]))
