# Intro
We are releasing the pre-trained low-rank AlexNet models described in our recent paper. 
We release the weights and all validation/loading code required to run the models. We will 
release the training scripts soon. The model weights are located in the 
[Box folder](https://ucmerced.box.com/s/gqtaucm2osjp5r7rlmk6qdcutrzcb6d6), and we additionally provide
the functionality to load the models directly:

```python
from model_def import low_rank_alexnet
m = low_rank_alexnet('scheme_2', 'v1', pretrained=True)
```

# Released models

Scheme-1 Low-Rank Models:

|                    |     MFLOPs     | top-1 err, %  |  top-5 err, %| FLOPs reduction |
| -------------      |:--------------:|:-------------:|:------------:|:---------------:|
|Caffe-AlexNet       |       724      |     42.70     |    19.80     |      1.00       |
|ours, v1            |       436      |   **41.56**   |  **19.15**   |      1.66       |
|ours, v2            |       263      |     42.63     |    19.95     |      2.75       |
|ours, v3            |       240      |     42.83     |    19.93     |    **3.01**     |


Scheme-2 Low-Rank Models:

|                    |     MFLOPs     | top-1 err, %  |  top-5 err, %| FLOPs reduction |
| -------------      |:--------------:|:-------------:|:------------:|:---------------:|
|Caffe-AlexNet       |       724      |     42.70     |    19.80     |      1.00       |
|ours, v1            |       324      |   **41.46**   |  **19.14**   |      2.23       |
|ours, v2            |       236      |     41.81     |    19.40     |      3.06       |
|ours, v3            |       190      |     42.07     |    19.54     |      3.81       |
|ours, v4            |     **151**    |     42.69     |    19.83     |    **4.79**     |

# Preprocessing, image transformations, data loading, etc.
Due to limited hardware resources, we trained our models on a single machine with a single GPU. Therefore, data 
loading/unloading of the ImageNet data was an issue and we did not use the default PyTorch mechanisms through 
torchvision. Instead, we used [Tensorpack](https://github.com/tensorpack/tensorpack/) library with some custom modules, 
and NVIDIA DALI for fast jpeg decoding on the GPU-s. 

### Dependencies
To recreate our data preprocessing pipeline, you will need to install the following dependencies (we recommend using Anaconda):

```bash
conda create -n imagenet_test python numpy scipy python-lmdb tqdm
conda activate imagenet_test
conda install pytorch==1.4.0 torchvision==0.5.0 cudatoolkit=10.0 -c pytorch
sudo apt-get install build-essential libcap-dev
pip install python-prctl
pip install tensorpack==0.9.8
pip install --extra-index-url https://developer.download.nvidia.com/compute/redist/cuda/9.0 nvidia-dali

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