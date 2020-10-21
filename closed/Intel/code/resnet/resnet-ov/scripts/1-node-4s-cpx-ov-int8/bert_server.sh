#!/bin/bash

export OV_MLPERF_BIN=</path/to/ov_mlperf>

${OV_MLPERF_BIN} --scenario Server \
	--mode Performance \
	--mlperf_conf </path/to/mlperf.conf> \
	--user_conf </path/to/bert/user.conf> \
	--model_name bert \
	--data_path </path/to/SQuAD-dataset> \ # path should contain vocab.txt, dev-v1.1.json
	--nireq 8 \
	--nthreads 112 \
	--nstreams 8 \
	--total_sample_count 10833 \
	--warmup_iters 1000 \
	--model_path </path/to/bert_int8.xml>

