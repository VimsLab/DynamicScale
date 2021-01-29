This code submission is to reproduce 3D Shape Classification results of ReducedPointNet++ oriented and scaled ellipsoid querying. <br>

Software requirements
=====================
Ubuntu 18.04<br>
Python 3.7.4 (recommend Anaconda3)<br>
CUDA 10.0 + cuDNN 7<br>
Cudatoolkit V10.0.130<br>
Tensorflow-gpu 1.14.0<br>

Download Code and Unzip
=======================
unzip paper23code.zip<br>
cd paper23code<br>

Download ModelNet40 Dataset
===========================
cd data<br>
wget https://shapenet.cs.stanford.edu/media/modelnet40_ply_hdf5_2048.zip --no-check-certificate<br>
unzip modelnet40_ply_hdf5_2048.zip<br>

Compiling cuda programs
=======================
cd ../tf_ops/sampling<br>
bash tf_sampling_compile.sh<br>
cd ../grouping<br>
bash tf_grouping_compile.sh<br>
cd ../3d_interpolation<br>
bash tf_interpolate_compile.sh<br>
cd ../../<br>

Training
=========
Run below command if you are using 2 GPUs for training (2x 11GB) - takes 4-5hours<br>
python train_multi_gpu.py<br>
Run below command if you are using only 1 GPU for training (1x 11GB) - takes 7-8 hours<br>
python train.py<br>

Testing
========
Run below command to test the model (takes 4 hours - edit line 115 to change evaluations)<br>
python evaluate.py --num_votes=12  <br>

Evaluation
==========
Change the log location in evaluate.py script from log to pretrained.<br>
Run below command to test through 100 evaluations and gives best results of 92.0% (takes 4 hours)<br>
python evaluate.py --num_votes=12  <br>

Note: This code has been heaviy borrowed from https://github.com/charlesq34/pointnet2 and from our previous work https://github.com/VimsLab/EllipsoidQuery<br>

To cite our paper please use below bibtex.<br>
  
```BibTex
        @InProceedings{Sheshappanavar_2021_MIPR,
            author = {Venkanna Sheshappanavar, Shivanand and Kambhamettu, Chandra},
            title = {Dynamic Local Geometry Capture in 3D Point Cloud Classification},
            booktitle = {Proceedings of the IEEE 4th International Conference on Multimedia Information Processing and Retrieval (IEEE MIPR 2021)},
            month = {September},
            year = {2021}
        }  
```
