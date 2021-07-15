# Intro
We are releasing the pre-trained low-rank AlexNet models described in our [CVPR2020 paper](https://openaccess.thecvf.com/content_CVPR_2020/html/Idelbayev_Low-Rank_Compression_of_Neural_Nets_Learning_the_Rank_of_Each_CVPR_2020_paper.html).
We release the weights and all validation/loading code required to run the models. 

## Update
We have found a mistake in data loading pipeline that severely limited the training and test accuracies of our AlexNet models. 
The problem was in casting the class's label: we inadvertently casted it to be uint8, basically limiting the nubmer of 
classes in the dataset to be only 256. On top of that, all classes with labels modulo 256 were grouped: e.g., classses 0, 
256, 512, and 768 were treated as class 0, and so on.

Upon fixing this bug, the accuracies of compressed models improved by 1.5% in average (see table below). All models weights
are updated to reflect the most recent training results, and can be downloaded as necessary using provided scripts.

We will release the training scripts soon. 


Scheme-1 Low-Rank Models:

|                           |     MFLOPs     | top-1 err, %  |  top-5 err, %| FLOPs reduction |
| ------------------------- |:--------------:|:-------------:|:------------:|:---------------:|
|Caffe-AlexNet              |       724      |     42.70     |    19.80     |      1.00       |
|ours, v1 (λ = 0.05×10⁻⁴)   |       436      |   **39.27**   |  **17.16**   |      1.66       |
|ours, v2 (λ = 0.15×10⁻⁴)   |       257      |     40.81     |    18.17     |      2.81       |
|ours, v3 (λ = 0.17×10⁻⁴)   |       248      |     41.11     |    18.36     |      2.92     |
|ours, v4 (λ = 0.20×10⁻⁴)   |       231      |     41.56     |    18.72     |    **3.13**     |

Scheme-2 Low-Rank Models:

|                            |     MFLOPs     | top-1 err, %  |  top-5 err, %| FLOPs reduction |
| -------------------------- |:--------------:|:-------------:|:------------:|:---------------:|
|Caffe-AlexNet               |       724      |     42.70     |    19.80     |      1.00       |
|ours, v1 (λ = 0.05×10⁻⁴)    |       321      |   **39.15**   |  **16.99**   |      2.25       |
|ours, v2 (λ = 0.10×10⁻⁴)    |       226      |     39.60     |    17.40     |      3.20       |
|ours, v3 (λ = 0.15×10⁻⁴)    |       185      |     39.93     |    17.47     |      3.92       |
|ours, v4 (λ = 0.20×10⁻⁴)    |       166      |     40.46     |    17.71     |      4.35       |
|ours, v5 (λ = 0.25×10⁻⁴)    |     **151**    |     41.03     |    18.23     |    **4.78**     |



# Loading the models
The model weights are standard PyTorch state_dict objects. Thus, you can load them as

```python
from model_def import low_rank_alexnet
m = low_rank_alexnet('scheme_2', 'v1', pretrained=True)
```
This will automatically download the models. If you would like to have the direct access to the models, see our 
[Box folder](https://ucmerced.box.com/s/gqtaucm2osjp5r7rlmk6qdcutrzcb6d6)

# Preprocessing, image transformations, data loading, etc.
Due to limited hardware resources, we trained our models on a single machine with a single GPU. Therefore, data 
loading/unloading of the ImageNet data was an issue and we did not use the default PyTorch mechanisms through 
torchvision. Instead, we used [Tensorpack](https://github.com/tensorpack/tensorpack/) library with some custom modules, 
and NVIDIA DALI for fast jpeg decoding on the GPU-s. 

### Dependencies
To recreate our data preprocessing pipeline, you will need to install the following dependencies (we recommend using Anaconda):

```bash
conda create -n imagenet_test python==3.8 numpy scipy python-lmdb tqdm
conda activate imagenet_test
conda install pytorch torchvision cudatoolkit=11.0 -c pytorch
sudo apt-get install build-essential libcap-dev
pip install python-prctl
pip install tensorpack==0.9.8
pip install --extra-index-url https://developer.download.nvidia.com/compute/redist/ nvidia-dali-cuda110==1.3.0
```

### Image resize, creating an LMDB database (validation only)

Once the previous script is finished, you will have an up-and-running conda environment called `imagenet_test`. 
In our pipeline, we resized the images outside of the training pipelines using ImageMagic. Now, assuming you have
ImageNet images stored in folder `~/imagenet` (which contains `val` and `train` folders), we need to preprocess the
 validation images to have smallest edge of size 256px:
```
chmod +x ./resize_images_val.sh
./resize_images_val.sh ~/imagenet/val
```
and then we are ready to create the LMDB database:

```bash
python3 build_lmdb.py --imagenet_folder ~/imagenet --val --lmdb_file ~/imagenet_val_256x.lmdb 

```
Once you have the validation lmdb file, you can verify the accuracy of our models by:
```
python3 model_def.py --lmdb_file ~/imagenet_val_256x.lmdb
```