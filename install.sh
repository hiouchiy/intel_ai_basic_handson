#!/bin/sh
conda create -n intel -c intel python=3.7 scikit-learn=0.22.1 pandas -y
conda create -n oss -c conda-forge python=3.7 scikit-learn=0.22.1 pandas -y
conda create -n intel_tf -c intel python=3.7 -y
conda activate intel_tf
pip install intel-tensorflow==2.2.0 pillow opencv-python pandas
conda create -n oss_tf -c conda-forge python=3.7 -y
conda activate oss_tf
pip install tensorflow==2.2.0 pillow opencv-python pandas
conda activate base
