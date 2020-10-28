#!/bin/sh

#################################################
# 1. Initial Parameter Check
#################################################

if [ $# -lt 1 ]; then
    echo "Please set physical core number as a first parameter of this script."
    echo "Use the command or utility tool to check the physical core number on your device."
    echo "---Linux: Launch a terminal -> type 'lscpu'"
	echo "---Windows: Launch Task Manager -> open 'Performance Tab'"
	echo "---MacOS: Launch a terminal -> type 'system_profiler SPHardwareDataType'"
else
#################################################
# 2. Install Intel Python
#################################################

	echo "Starting installing Intel python and OSS python into each virtual environment..."

	#Create Intel Python virtual environment on Anaconda
	conda create -n intel -c intel python=3.7 scikit-learn=0.22.1 pandas -y

	#Create OSS Python virtual environment on Anaconda
	conda create -n oss -c conda-forge python=3.7 scikit-learn=0.22.1 pandas -y

	#Set environemnt variable 'USE_DAAL4PY_SKLEARN=YES' for accelaration of Intel Scikit-learn
	export USE_DAAL4PY_SKLEARN=YES

	echo "Installing Intel python and OSS python is done."

#################################################
# 3. Install Intel TensorFlow
#################################################

	echo "Starting installing Intel TensorFlow and OSS TensorFlow into each virtual environment..."
	echo "Physical core number $1 is going to be set as environment parameter 'OMP_NUM_THREADS'."

	#Create a virtual environment and install Intel TensorFlow there
	conda create -n intel_tf -c intel python=3.7 -y
	conda activate intel_tf
	pip install intel-tensorflow==2.2.0 pillow opencv-python pandas

	#Create a virtual environment and install OSS TensorFlow there
	conda create -n oss_tf -c conda-forge python=3.7 -y
	conda activate oss_tf
	pip install tensorflow==2.2.0 pillow opencv-python pandas

	#Set some environment variable for accelarating Intel TensorFlow
	export KMP_AFFINITY=granularity=fine,compact,1,0
	export KMP_BLOCKTIME=1
	export KMP_SETTINGS=1
	export OMP_NUM_THREADS=$1
	
	#Download training data and unzip it.
	wget https://hiouchiystorage.blob.core.windows.net/share/data.zip
	mkdir train_data
	mv data.zip train_data/
	unzip train_data/data.zip -d train_data/

	echo "Installing Intel TensorFlow and OSS TensorFlow is done."

	#Go back to 'base' environemnt
	conda activate base	
fi

