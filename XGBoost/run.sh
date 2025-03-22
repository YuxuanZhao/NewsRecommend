#!/bin/bash

python3 faiss_gpu_clustering.py
python3 xgboost_ranking.py
python3 xgboost_inferencing.py