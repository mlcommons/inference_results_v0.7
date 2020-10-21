"""
mlperf inference benchmarking tool
"""

from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import argparse
import array
import collections
import json
import logging
import os
import sys
import threading
import time
from queue import Queue
import multiprocessing
import subprocess

import mlperf_loadgen as lg
import numpy as np

import dataset
import imagenet

logging.basicConfig(level=logging.INFO)
log = logging.getLogger("main")

in_queue_cnt = 0
out_queue_cnt = 0

# will be override via args
num_phy_cpus = 56
num_ins = 14

# the datasets we support
SUPPORTED_DATASETS = {
    "imagenet":
        (imagenet.Imagenet, dataset.pre_process_vgg, dataset.PostProcessArgMax(offset=-1),
         {"image_size": [224, 224, 3]}),
}

# pre-defined command line options so simplify things. They are used as defaults and can be
# overwritten from command line
SUPPORTED_PROFILES = {
    "defaults": {
        "dataset": "imagenet",
        "backend": "mxnet",
        "cache": 0,
        "max-batchsize": 32,
    },

    # resnet
    "resnet50-tf": {
        "inputs": "input_tensor:0",
        "outputs": "ArgMax:0",
        "dataset": "imagenet",
        "backend": "tensorflow",
        "model-name": "resnet50",
    },
    "resnet50-onnxruntime": {
        "dataset": "imagenet",
        "outputs": "ArgMax:0",
        "backend": "onnxruntime",
        "model-name": "resnet50",
    },
}

SCENARIO_MAP = {
    "SingleStream": lg.TestScenario.SingleStream,
    "MultiStream": lg.TestScenario.MultiStream,
    "Server": lg.TestScenario.Server,
    "Offline": lg.TestScenario.Offline,
}


def get_args():
    """Parse commandline."""
    parser = argparse.ArgumentParser()
    parser.add_argument('--symbol-file', type=str, required=True, help='symbol file path')
    parser.add_argument('--param-file', type=str, required=False, help='param file path')
    parser.add_argument('--label-name', type=str, default='softmax_label', help='label name')
    parser.add_argument('--batch-size', type=int, default=1, help='batch size')
    parser.add_argument("--num-instance", default=2, type=int, help="number of instance")
    parser.add_argument("--num-phy-cpus", default=56, type=int, help="number of pyhsical cpu cores")
    parser.add_argument("--dataset-path", required=True, help="path to the dataset")
    parser.add_argument("--dataset-list", help="path to the dataset list")
    parser.add_argument("--scenario", default="SingleStream",
                        help="mlperf benchmark scenario, one of " + str(list(SCENARIO_MAP.keys())))
    parser.add_argument("--output-file", type=str, default='accuracy.txt', help="test results")
    parser.add_argument("--inputs", help="model inputs")
    parser.add_argument("--outputs", help="model outputs")
    parser.add_argument("--qps", type=int, help="target qps")
    parser.add_argument("--cache", type=int, default=0, help="use cache")
    parser.add_argument("--cache-dir", type=str, default=None, help="cache directory")
    parser.add_argument("--accuracy", action="store_true", help="enable accuracy pass")
    parser.add_argument("--use-int8-dataset", default=False, action="store_true", help="whether to use int8 dataset")
    parser.add_argument("--find-peak-performance", action="store_true", help="enable finding peak performance pass")

    # file to use mlperf rules compliant parameters
    parser.add_argument("--mlperf-conf", type=str, default="mlperf.conf", help="mlperf rules config")
    # file for user LoadGen settings such as target QPS
    parser.add_argument("--user-conf", type=str, default="user.conf", help="user config for user LoadGen settings such as target QPS")

    # below will override mlperf rules compliant settings - don't use for official submission
    parser.add_argument("--count", default=None, type=int, help="dataset items to use")
    args = parser.parse_args()

    if args.inputs:
        args.inputs = args.inputs.split(",")
    if args.outputs:
        args.outputs = args.outputs.split(",")

    if args.scenario not in SCENARIO_MAP:
        parser.error("valid scanarios:" + str(list(SCENARIO_MAP.keys())))
    return args


def block_until(counter, num_ins, t):
    while counter.value < num_ins:
        time.sleep(t)


class QueueRunner():
    def __init__(self, task_queue, batch_size):
        self.task_queue = task_queue
        self.batch_size = batch_size
        self.query_id_list = []
        self.sample_index_list = []
        self.index = 0

    def put(self, query_ids, sample_indexes):
        global in_queue_cnt

        idx = query_ids if isinstance(query_ids, list) else [query]
        sample_id = sample_indexes if isinstance(sample_indexes, list) else [sample_indexes]

        num_samples = len(sample_id)
        if num_samples < self.batch_size:
            self.index += 1
            if self.index < self.batch_size:
                self.query_id_list.append(idx[0])
                self.sample_index_list.append(sample_id[0])
            else:
                self.query_id_list.append(idx[0])
                self.sample_index_list.append(sample_id[0])
                self.task_queue.put(Input(self.query_id_list, self.sample_index_list))
                in_queue_cnt += self.batch_size
                self.index = 0
                self.query_id_list = []
                self.sample_index_list = []
        else:
            bs = self.batch_size
            for i in range(0, len(idx), bs):
                ie = i + bs
                self.task_queue.put(Input(idx[i:ie], sample_id[i:ie]))
                in_queue_cnt += len(sample_id[i:ie])
        #print ('in_queue_cnt=', in_queue_cnt)


class Input(object):
    def __init__(self, id_list, index_list):
        assert isinstance(id_list, list)
        assert isinstance(index_list, list)
        assert len(id_list) == len(index_list)
        self.query_id_list = id_list
        self.sample_index_list = index_list


class Output(object):
    def __init__(self, query_id_list, result, good, total, take_accuracy=False):
        self.query_id_list = query_id_list
        self.result = result
        self.good = good
        self.total = total
        self.take_accuracy = take_accuracy


class Consumer(multiprocessing.Process):
    def __init__(self, task_queue, result_queue, ds_queue, lock, init_counter, proc_idx, args):
        multiprocessing.Process.__init__(self)
        global num_ins
        global num_phy_cpus

        self.task_queue = task_queue
        self.result_queue = result_queue
        self.ds_queue = ds_queue
        self.lock = lock
        self.init_counter = init_counter
        self.proc_idx = proc_idx
        self.args = args
        self.affinity = range(round(proc_idx * num_phy_cpus / num_ins),
                              round((proc_idx + 1) * num_phy_cpus / num_ins))
        self.start_core_idx = proc_idx * num_phy_cpus // num_ins
        self.end_core_idx = (proc_idx + 1) * num_phy_cpus // num_ins - 1
        self.data_shape = (self.args.batch_size, 3, 224, 224)
        self.label_shape = (self.args.batch_size,)
        self.post_proc = None
        self.num_cores_per_instance = num_phy_cpus // num_ins

    def load_model(self):
        import mxnet as mx
        ctx = mx.cpu()
        symbol_file_path = self.args.symbol_file
        log.info('Loading symbol from file %s' % symbol_file_path)
        symbol = mx.sym.load(symbol_file_path)
        param_file_path = self.args.param_file
        log.info('Loading params from file %s' % param_file_path)
        save_dict = mx.nd.load(param_file_path)
        arg_params = {}
        aux_params = {}
        for k, v in save_dict.items():
            tp, name = k.split(':', 1)
            if tp == 'arg':
                arg_params[name] = v
            if tp == 'aux':
                aux_params[name] = v

        mod = mx.mod.Module(symbol=symbol, context=ctx, label_names=[self.args.label_name, ])
        if self.args.use_int8_dataset:
            dshape = mx.io.DataDesc(name='data', shape=self.data_shape, dtype=np.int8)
        else:
            dshape = mx.io.DataDesc(name='data', shape=self.data_shape, dtype=np.float32)
        label_shapes = [('softmax_label', self.label_shape)]
        mod.bind(for_training=False,
                 data_shapes=[dshape],
                 label_shapes=label_shapes)
        mod.set_params(arg_params, aux_params)
        return mod

    def load_dataset(self):
        image_format = 'NCHW'
        dataset = "imagenet"
        wanted_dataset, pre_proc, post_proc, kwargs = SUPPORTED_DATASETS[dataset]
        self.post_proc = post_proc
        self.post_proc.start()
        ds = wanted_dataset(data_path=self.args.dataset_path,
                            image_list=self.args.dataset_list,
                            name=dataset,
                            image_format=image_format,
                            pre_process=pre_proc,
                            use_cache=self.args.cache,
                            cache_dir=self.args.cache_dir,
                            count=self.args.count,
                            num_workers=self.num_cores_per_instance,
                            use_int8=self.args.use_int8_dataset,
                            **kwargs)
        return ds

    def warmup(self, mod, ds):
        import mxnet as mx
        ctx = mx.cpu()
        sample_list = list(range(self.args.batch_size))
        ds.load_query_samples(sample_list)

        data_np, label_np = ds.get_samples(sample_list)
        data = mx.nd.array(data_np, dtype=data_np.dtype).as_in_context(ctx)
        label = mx.nd.array(label_np, dtype=label_np.dtype).as_in_context(ctx)
        batch = mx.io.DataBatch([data], [label])

        for _ in range(5):
            mod.forward(batch, is_train=False)
            mx.nd.waitall()
        ds.unload_query_samples(sample_list)

    def run(self):
        os.sched_setaffinity(self.pid, self.affinity)
        affinity = os.sched_getaffinity(self.pid)
        log.info('Process {}, affinity proc list:{}'.format(self.pid, affinity))
        import mxnet as mx
        ctx = mx.cpu()

        # load mxnet model
        mod = self.load_model()
        ds = self.load_dataset()

        # warmup
        self.warmup(mod, ds)

        # Inform master process to continue till sample_list was issue
        self.lock.acquire()
        self.init_counter.value += 1
        self.lock.release()

        sample_list = self.ds_queue.get()
        ds.load_query_samples(sample_list)
        # Inform master process to continue loadgen startTest
        self.lock.acquire()
        self.init_counter.value += 1
        self.lock.release()

        while True:
            next_task = self.task_queue.get()
            if next_task == 'DONE':
                log.info('Exiting {}-pid:{}'.format(self.name, self.pid))
                ds.unload_query_samples(sample_list)
                self.task_queue.task_done()
                break

            processed_results = []
            query_id_list = next_task.query_id_list
            sample_index_list = next_task.sample_index_list
            batch_size = len(sample_index_list)

            data_np, label_np = ds.get_samples(sample_index_list)
            data = mx.nd.array(data_np, dtype=data_np.dtype).as_in_context(ctx)
            label = mx.nd.array(label_np, dtype=label_np.dtype).as_in_context(ctx)
            batch = mx.io.DataBatch([data], [label])

            mod.forward(batch, is_train=False)
            out = mod.get_outputs()
            processed_results = self.post_proc(results=out, ids=None, expected=label_np)

            result = Output(query_id_list, processed_results, self.post_proc.good, self.post_proc.total, take_accuracy=self.args.accuracy)
            self.result_queue.put(result)
            self.task_queue.task_done()


def response_loadgen(out_queue):
    global out_queue_cnt

    while True:
        next_task = out_queue.get()
        if next_task == 'DONE':
            log.info('Exiting response thread')
            break
        query_id_list = next_task.query_id_list
        result = next_task.result

        response_array_refs = []
        response = []
        for idx, query_id in enumerate(query_id_list):
            response_array = array.array("B", np.array(result[idx], np.float32).tobytes())
            response_array_refs.append(response_array)
            bi = response_array.buffer_info()
            response.append(lg.QuerySampleResponse(query_id, bi[0], bi[1]))
        lg.QuerySamplesComplete(response)
        out_queue_cnt += len(query_id_list)


def main():
    global num_ins
    global num_phy_cpus
    global in_queue_cnt
    global out_queue_cnt

    args = get_args()
    log.info(args)

    num_ins = args.num_instance
    num_phy_cpus = args.num_phy_cpus
    log.info('Run with {} instance on {} cpus'.format(num_ins, num_phy_cpus))

    mlperf_conf = os.path.abspath(args.mlperf_conf)
    if not os.path.exists(mlperf_conf):
        log.error("{} not found".format(mlperf_conf))
        sys.exit(1)

    user_conf = os.path.abspath(args.user_conf)
    if not os.path.exists(user_conf):
        log.error("{} not found".format(user_conf))
        sys.exit(1)

    image_format = 'NCHW'
    dataset = "imagenet"
    wanted_dataset, pre_proc, post_proc, kwargs = SUPPORTED_DATASETS[dataset]

    ds = wanted_dataset(data_path=args.dataset_path,
                        image_list=args.dataset_list,
                        name=dataset,
                        image_format=image_format,
                        pre_process=pre_proc,
                        use_cache=args.cache,
                        cache_dir=args.cache_dir,
                        count=args.count,
                        use_int8=args.use_int8_dataset,
                        num_workers=num_phy_cpus,
                        **kwargs)

    # Establish communication queues
    log.info('Start comsumer queue and response thread')
    lock = multiprocessing.Lock()
    init_counter = multiprocessing.Value("i", 0)
    in_queue = multiprocessing.JoinableQueue()
    out_queue = multiprocessing.Queue()
    ds_queue = multiprocessing.Queue()

    # Start consumers
    consumers = [Consumer(in_queue, out_queue, ds_queue, lock, init_counter, i, args)
                 for i in range(num_ins)]
    for c in consumers:
        c.start()

    # Wait until all sub-processors are ready
    block_until(init_counter, num_ins, 2)

    # Start response thread
    response_worker = threading.Thread(
        target=response_loadgen, args=(out_queue,))
    response_worker.daemon = True
    response_worker.start()

    scenario = SCENARIO_MAP[args.scenario]
    runner = QueueRunner(in_queue, args.batch_size)

    def issue_queries(response_ids, query_sample_indexes):
        runner.put(response_ids, query_sample_indexes)

    def flush_queries():
        pass

    def process_latencies(latencies_ns):
        log.info("Average latency: {}".format(np.mean(latencies_ns)))
        log.info("Median latency: {}".format(np.percentile(latencies_ns, 50)))
        log.info("90 percentile latency: {}".format(np.percentile(latencies_ns, 90)))

    def load_query_samples(sample_list):
        for _ in range(num_ins):
            ds_queue.put(sample_list)
        block_until(init_counter, 2 * num_ins, 2)

    def unload_query_samples(sample_list):
        pass

    settings = lg.TestSettings()
    settings.FromConfig(mlperf_conf, "resnet50", args.scenario)
    settings.FromConfig(user_conf, "resnet50", args.scenario)
    settings.scenario = scenario
    settings.mode = lg.TestMode.PerformanceOnly
    if args.accuracy:
        settings.mode = lg.TestMode.AccuracyOnly
    if args.find_peak_performance:
        settings.mode = lg.TestMode.FindPeakPerformance

    if args.qps:
        qps = float(args.qps)
        settings.server_target_qps = qps
        settings.offline_expected_qps = qps

    count = ds.get_item_count()
    perf_count = 1024
    if args.accuracy:
        perf_count = count
    sut = lg.ConstructFastSUT(issue_queries, flush_queries, process_latencies)
    qsl = lg.ConstructQSL(count, perf_count, load_query_samples, unload_query_samples)

    log.info("starting {}".format(scenario))
    lg.StartTest(sut, qsl, settings)

    # Wait until outQueue done
    while out_queue_cnt < in_queue_cnt:
        time.sleep(0.2)

    in_queue.join()
    for i in range(num_ins):
        in_queue.put('DONE')
    for c in consumers:
        c.join()
    out_queue.put('DONE')

    if args.accuracy:
        output_file = 'accuracy.txt'
        if args.output_file:
            output_file = args.output_file
        cmd = "python tools/accuracy-imagenet.py " \
              "--mlperf-accuracy-file=mlperf_log_accuracy.json " \
              "--imagenet-val-file=val_map.txt --output-file={}".format(output_file)
        cmd = cmd.split(' ')
        subprocess.check_call(cmd)

    lg.DestroyQSL(qsl)
    lg.DestroyFastSUT(sut)

    log.info('Test done.')


if __name__ == "__main__":
    main()
