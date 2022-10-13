# [Task Adaptive Parameter Sharing for Multi-Task Learning](https://arxiv.org/abs/2203.16708)

Unofficial Pytorch implementation of **Task Adaptive Parameter Sharing** (CVPR 2022). <br />


<p align="center">
<img src="./assets/teaser.jpg" width="512"/>
</p>

Task Adaptive Parameter Sharing (TAPS) is a general method for tuning a base model to a new task by adaptively modifying a small, task-specific subset of layers. This enables multi-task learning while minimizing resources used and competition between tasks. TAPS solves a joint optimization problem which determines which layers to share with the base model and the value of the task-specific weights.


## Installation

### Requirements

### Datasets

**ImageNet-to-Sketch**
The 5 datasets comprising ImagetNet-to-Sketch can be download from the [PiggyBack repository](https://github.com/arunmallya/piggyback) at this link: [https://uofi.box.com/s/ixncr3d85guosajywhf7yridszzg5zsq](https://uofi.box.com/s/ixncr3d85guosajywhf7yridszzg5zsq)

Place the datasets in the datasets folder.

## Training

### Training Arguments
Relevant command line arguments that you may want to adjust. For the full list of options see options.py. Arguments for experiments are logged in settings.txt in their respective folder. 

```

--
--lam - The sparsity coefficient. Larger lam results in fewer layers being tuned (λ in the paper).
--dataset - relative path to dataset
--model_type - Network architecture. Currently supports ResNet34 and ResNet50. Support for VIT and all convolution networks coming soon. 
--lr - Learning rate 
--model_path - Relative path to a pretrained model. Default option uses the pytorch pretrained models.
```

### Sequential TAPS Training
Train a pretrained network with TAPS on the sketch dataset. 
```
python train_sequential.py --dataset ../datasets/DomainNet/sketch --experiment_name ./results/sketch --multi_gpu --model_type resnet34
```



### Joint TAPS Training


## Evaluation

### Tensorboard

### Visualizing Modified Layers
