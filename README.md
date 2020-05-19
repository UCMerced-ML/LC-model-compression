`LC-model-compression` is a flexible, extensible software framework that allows a user to do optimal compression, with minimal effort, of a neural network or other machine learning model using different compression schemes. It is based on the [Learning-Compression (LC) algorithm](http://arxiv.org/abs/1707.01209), which performs an iterative optimization of the compressed model by alternating a *learning (L) step* with a *compression (C) step*. This decoupling of the "machine learning" and "signal compression" aspects of the problem make it possible to use a common optimization and software framework to handle any choice of model and compression scheme; all that is needed to compress model X with compression Y is to call the corresponding algorithms in the L and C steps, respectively. The software fully supports this by design, which makes it flexible and extensible. A number of neural networks and compression schemes are currently supported, and we expect to add more in the future. These include neural networks such as LeNet, ResNet, VGG, NiN, etc. (as well as linear models); and compression schemes such as low-rank and tensor factorization (including automatically learning the layer ranks), various forms of pruning and quantization, and combinations of all of those. For a neural network, the user can choose different compression schemes for different parts of the network.

The LC algorithm is efficient in runtime; it does not take much longer than training the reference, uncompressed model in the first place. The compressed models perform very competitively and allow the user to easily explore the space of prediction accuracy of the model vs compression ratio (which can be defined in terms of memory, inference time, energy or other criteria).

`LC-model-compression` is written in Python and PyTorch, and has been extensively tested since 2017 in research projects at UC Merced. You can find some of these in the examples below, or in our [papers about the LC algorithm](https://faculty.ucmerced.edu/mcarreira-perpinan/research/LC-model-compression.html).
  
## Features
`LC-model-compression` supports various compression schemes and allows the user to combine them in a mix-and-match way. Some examples: 
- a single compression per layer (e.g. low-rank compression for layer 1 with maximum rank 5)  
- a single compression over multiple layers (e.g. prune 5\% of weights in layer 1 and 3, jointly)  
- mixing multiple compressions (e.g. quantize layer 1 and prune jointly layers 2 and 3)  
- additive combinations of compressions (e.g. represent a layer as a quantized value with an additive sparse correction)  
  
At present, we support the following compression schemes:

| Scheme        |  Formulation  | LC-model-compression Class  |  
| ------------- |:--------------|-----------------------------|  
| Quantization  | Adaptive quantization (with learned codebook) <br> Binarization into {-1, 1} and {-c, c} <br> Ternarization into \{-c, 0, c\} | `AdaptiveQuantization` <br> `BinaryQuantization, ScaledBinaryQuantization` <br> `ScaledTernaryQuantization` |  
| Pruning       | l<sub>0</sub>/l<sub>1</sub> constraint pruning <br> l<sub>0</sub>/l<sub>1</sub> penalty pruning                                      |  `ConstraintL0Pruning`, `ConstraintL1Pruning` <br> `PenaltyL0Pruning`, `PenaltyL1Pruning`  |  
| Low-rank      | Low-rank compression to a given rank  <br> Low-rank with automatic rank selection      |  `LowRank` <br> `RankSelection` |  

## Examples of use
If you want to compress your own models, you can use the following examples as a guide:
- Compressing the [LeNet300 neural network](examples/lenet300/README.md).
- Reproducing the results of our [CVPR2018 paper](examples/cvpr2018/README.md) on pruning (coming soon)
- Reproducing the results of our [CVPR2020 paper](examples/cvpr2020/README.md) on low-rank compression with rank selection (coming soon)
  
## Installation  
We recommend installing the dependencies through [conda](https://conda.io) into a new environment:  
```  
conda create -n lc_package python==3.7  
conda install -n lc_package numpy==1.16.2 scipy==1.2.1 scikit-learn==0.20.2  
```
You will need to install PyTorch v1.1 to the same conda environment. In principle, newer PyTorch versions should work to, however we have not fully tested them. The specific installation instruction might differ from system to system, confirm with [official site](https://pytorch.org/get-started/previous-versions/). On our system we used following:
```bash
conda install -n lc_package pytorch==1.1.0 torchvision==0.3.0 cudatoolkit=10.0 -c pytorch 
```
Once the requirements are installed, within the conda environment:
```
conda activate lc_package
git clone https://github.com/UCMerced-ML/LC-model-compression
pip install -e ./LC-model-compression
```

## Citation
If you find this code useful, please cite it as:
```
Yerlan Idelbayev and Miguel Á. Carreira-Perpiñán: 
"A flexible, extensible software framework for model compression based on the LC algorithm".
arXiv:2005.07786, May 15, 2020.
http://arxiv.org/abs/2005.07786
```
