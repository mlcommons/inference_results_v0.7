#!/bin/bash
# Forwards all arguments to the rnnt harness. Very bad WAR, but using make is too much typing
# Uses AccuracyOnly test_mode
./build/bin/harness_rnnt "${@:1}" --test_mode=AccuracyOnly && python3 code/rnnt/tensorrt/accuracy.py
