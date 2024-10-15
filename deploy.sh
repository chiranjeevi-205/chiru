#!/usr/bin/bash
cd /home/eizen/Desktop/Edge_devices/Lset-Attention-Inference
source /home/eizen/Desktop/Edge_devices/Lset-Attention-Inference/venv/bin/activate

export PYTHONPATH=$PWD

/home/eizen/Desktop/Edge_devices/Lset-Attention-Inference/venv/bin/python3 /home/eizen/Desktop/Edge_devices/Lset-Attention-Inference/api_endpoint.py &

/home/eizen/Desktop/Edge_devices/Lset-Attention-Inference/venv/bin/python3 /home/eizen/Desktop/Edge_devices/Lset-Attention-Inference/s.py



