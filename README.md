This code submission is to reproduce 3D Shape Classification results of ReducedPointNet++ oriented and scaled ellipsoid querying. 

Software requirements
=====================
Ubuntu 18.04
Python 3.7.4 (recommend Anaconda3)
CUDA 10.0 + cuDNN 7
Cudatoolkit V10.0.130
Tensorflow-gpu 1.14.0

Download Code and Unzip
=======================
unzip paper23code.zip
cd paper23code

Download ModelNet40 Dataset
===========================
cd data
wget https://shapenet.cs.stanford.edu/media/modelnet40_ply_hdf5_2048.zip --no-check-certificate
unzip modelnet40_ply_hdf5_2048.zip

Compiling cuda programs
=======================
cd ../tf_ops/sampling
bash tf_sampling_compile.sh
cd ../grouping
bash tf_grouping_compile.sh
cd ../3d_interpolation
bash tf_interpolate_compile.sh
cd ../../

Training
=========
Run below command if you are using 2 GPUs for training (2x 11GB) - takes 4-5hours
python train_multi_gpu.py
Run below command if you are using only 1 GPU for training (1x 11GB) - takes 7-8 hours
python train.py

Testing
========
Run below command to test the model (takes 4 hours - edit line 115 to change evaluations)
python evaluate.py --num_votes=12  

Evaluation
==========
Change the log location in evaluate.py script from log to pretrained.
Run below command to test through 100 evaluations and gives best results of 92.0% (takes 4 hours)
python evaluate.py --num_votes=12  

Note: This code has been heaviy borrowed from https://github.com/charlesq34/pointnet2 and from our previous work https://github.com/VimsLab/EllipsoidQuery
