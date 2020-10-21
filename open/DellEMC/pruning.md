# Xilinx MLPerf Pruning

There are two main methods of pruning. The simplest form of pruning is called "fine-grained pruning" and results in sparse weight matrices. There is also "coarse-grained pruning", which eliminates neurons that do not contribute significantly to the network's accuracy. 

We employ the "coarse-grained pruning" method. For convolutional layers, "coarse-grained pruning" prunes the entire 3D kernel, so it is also called channel pruning. We use L1-norm to select unimportant channels and physically prune them. Retraining is used to adjust the remaining weights to recover accuracy.

We use the full training set of ImageNet and retrained the pruned model for 30 epochs to recover the accuracy. The total NW-level pruning is 25%. 
