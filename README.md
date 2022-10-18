# Large scale training on PM and DRAM
Using Pytorch Geometric (PyG) to implement GCN models (full-batch), GraphSAGE models (with sampling), ClusterGCN models (with sampling) on Cora, Citeseer, Pubmed and Reddit datasets
Implemented and used MMAP and DirectIO based data loaders and combined with a Python front end.

Third party libraries:

+ Pytorch
+ Pytorch Geometric（PyG）
+ nvsmi
+ threading
+ psutil
+ directio
+ ctypes

Project structure:

+ data ----------------------------- Dictionary storing the dataset
+ figure --------------------------- Dictionary storing the output graph
+ System evaluation metric ----Dictionary contains the code of system metric collection function and basic GCN model training
+ mmaploader.c    mmaploader.so ---- C source and compiled files
+ MP.py  SAGEConv.py  ------------   Customized SageConv (adds the ability to return the time of each step during training)
+ SAGEWithCluster.ipynb ------------------- ClusterGCN experiments
+ SAGEWithoutCluster.ipynb ------------------- GraphSage experiments
