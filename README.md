# DeformSyncNet
Pytorch implementation of  of DeformSyncNet: Deformation Transfer via Synchronized Shape Deformation Spaces (SIGGRAPH Asia 2020)

## Citation



## Introduction



## Dependencies

- Python 3.7 (Recommend to use [Anaconda](https://www.anaconda.com/download/#linux))
- [PyTorch 1.3.1](https://pytorch.org/)
- NVIDIA GPU + [CUDA 10.0](https://developer.nvidia.com/cuda-downloads)
- Python packages: `pip install tqdm matplotlib tensorboardX pyyaml h5py`
- Submodule [ThibaultGROUEIX/ChamferDistancePytorch](https://github.com/ThibaultGROUEIX/ChamferDistancePytorch): run `git submodule update --init --recursive`

## Dataset Preparation

### Download data

#### ShapeNet 

**Full raw data(train, val and test)** can be downloaded with `wget --no-check-certificate https://shapenet.cs.stanford.edu/media/minhyuk/DeformSyncNet/data/ShapeNetFullData.zip`. Please download and unzip the `ShapeNetFullData.zip` file.

**Prepared test data** can be downloaded with `wget --no-check-certificate https://shapenet.cs.stanford.edu/media/minhyuk/DeformSyncNet/data/ShapeNetTestData.zip`. Please download and unzip the `ShapeNetTestData.zip` file.

#### ComplementMe 

**Full raw data(train, val and test)** can be downloaded with `wget --no-check-certificate https://shapenet.cs.stanford.edu/media/minhyuk/DeformSyncNet/data/ComplementMeFullData.zip`. Please download and unzip the `ComplementMeFullData.zip` file.

**Prepared test data** can be downloaded with `wget --no-check-certificate https://shapenet.cs.stanford.edu/media/minhyuk/DeformSyncNet/data/ComplementMeTestData.zip`. Please download and unzip the `ComplementMeTestData.zip` file.

## Training

To train a model:
```python
cd code
python train.py -opt option/train/train_DSN_(ShapeNet|ComplementMe)_{category}.yaml
```

- The json file will be processed by `option/parse.py`. Please refer to [this](./code/option/train/README.md) for more details.
- Before running this code, please modify option files to your own configurations including:
  - proper `root` path for the data loader
  - saving frequency for models and states
  - other hyperparameters
  - loss function, etc. 
- During training, you can use Tesorboard to monitor the losses with
`tensorboard --logdir tb_logger/NAME_OF_YOUR_EXPERIMENT`

## Testing
To test trained model with metrics in Table 1(Fitting CD, MIOU, MMD-CD, Cov-CD) and Table2(Parallelogram consistency CD) (on ShapeNet) in the paper:

```python
cd code
python test.py -opt path/to/train_option -test_data_root path/to/test_data -data_root path/to/full/data -out_dir path/to/save_dir -load_path path/to/model
```

To test trained model with metrics in Table 3(Fitting CD, MMD-CD, Cov-CD) (on ComplementMe) in the paper:

```python
cd code
python test_ComplementMe.py -opt path/to/train_option -test_data_root path/to/test_data -out_dir path/to/save_dir -load_path path/to/model
```

It will load model weight from `path/to/model`. The default loading directory is `experiment/{exp_name}/model/best_model.pth`, which means when you test model after training, you can omit the `-load_path`. Generated shapes will be save in `path/to/save_dir`. The default save directory is `result/ShapeNet/{category}`.
