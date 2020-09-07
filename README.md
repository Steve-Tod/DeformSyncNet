# DeformSyncNet: Deformation Transfer via Synchronized Shape Deformation Spaces

[Minhyuk Sung](http://mhsung.github.io/)\*, [Zhenyu Jiang](http://jiangzhenyu.xyz)\*, [Panos Achlioptas](http://ai.stanford.edu/~optas/), [Niloy J. Mitra](http://www0.cs.ucl.ac.uk/staff/n.mitra/), [Leonidas J. Guibas](https://geometry.stanford.edu/member/guibas/) (\* equal contribution)

SIGGRAPH Asia 2020 (To appear)

[Project](https://mhsung.github.io/deform-sync-net.html) | [arxiv](https://arxiv.org/abs/2009.01456)

## Citation

```bibtex
@article{Sung:2020,
  author = {Sung, Minhyuk and Jiang, Zhenyu and Achlioptas, Panos and Mitra, Niloy J. and Guibas, Leonidas J.},
  title = {DeformSyncNet: Deformation Transfer via Synchronized Shape Deformation Spaces},
  Journal = {ACM Transactions on Graphics (Proc. of SIGGRAPH Asia)}, 
  year = {2020}
}
```

## Introduction

Shape deformation is an important component in any geometry processing toolbox. The goal is to enable intuitive deformations of single or multiple shapes or to transfer example deformations to new shapes while preserving the plausibility of the deformed shape(s). Existing approaches assume access to point-level or part-level correspondence or establish them in a preprocessing phase, thus limiting the scope and generality of such approaches. We propose DeformSyncNet, a new approach that allows consistent and synchronized shape deformations without requiring explicit correspondence information. Technically, we achieve this by encoding deformations into a class-specific idealized latent space while decoding them into an individual, model-specific linear deformation action space, operating *directly* in 3D. The underlying encoding and decoding are performed by specialized (jointly trained) neural networks. By design, the inductive bias of our networks results in a deformation space with several desirable properties, such as path invariance across different deformation pathways, which are then also approximately preserved in real space. We qualitatively and quantitatively evaluate our framework against multiple alternative approaches and demonstrate improved performance.

## Dependencies

- Python 3.7 (Recommend to use [Anaconda](https://www.anaconda.com/download/#linux))
- [PyTorch 1.3.1](https://pytorch.org/)
- NVIDIA GPU + [CUDA 10.0](https://developer.nvidia.com/cuda-downloads)
- Python packages: `pip install tqdm matplotlib tensorboardX pyyaml h5py`
- Submodule [ThibaultGROUEIX/ChamferDistancePytorch](https://github.com/ThibaultGROUEIX/ChamferDistancePytorch): run `git submodule update --init --recursive`

## Dataset Preparation

### Download data

#### ShapeNet 

**Full raw data(train, val and test)** can be downloaded [here](https://shapenet.cs.stanford.edu/media/minhyuk/DeformSyncNet/data/ShapeNetFullData.zip)(you can use `wget --no-check-certificate {url}` to download in commandline). Please download and unzip the `ShapeNetFullData.zip` file.

**Prepared test data** can be downloaded [here](https://shapenet.cs.stanford.edu/media/minhyuk/DeformSyncNet/data/ShapeNetTestData.zip)(you can use `wget --no-check-certificate {url}` to download in commandline). Please download and unzip the `ShapeNetTestData.zip` file.

#### ComplementMe 

**Full raw data(train, val and test)** can be downloaded [here](https://shapenet.cs.stanford.edu/media/minhyuk/DeformSyncNet/data/ComplementMeFullData.zip)(you can use `wget --no-check-certificate {url}` to download in commandline). Please download and unzip the `ComplementMeFullData.zip` file

**Prepared test data** can be downloaded [here](https://shapenet.cs.stanford.edu/media/minhyuk/DeformSyncNet/data/ComplementMeTestData.zip)(you can use `wget --no-check-certificate {url}` to download in commandline). Please download and unzip the `ComplementMeTestData.zip` file.

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

## Pretrained Models

### ShapeNet

[Airplane](https://shapenet.cs.stanford.edu/media/minhyuk/DeformSyncNet/data/models/DSN_ShapeNet_Airplane.pth), [Car](https://shapenet.cs.stanford.edu/media/minhyuk/DeformSyncNet/data/models/DSN_ShapeNet_Car.pth), [Chair](https://shapenet.cs.stanford.edu/media/minhyuk/DeformSyncNet/data/models/DSN_ShapeNet_Chair.pth), [Lamp](https://shapenet.cs.stanford.edu/media/minhyuk/DeformSyncNet/data/models/DSN_ShapeNet_Lamp.pth), [Table](https://shapenet.cs.stanford.edu/media/minhyuk/DeformSyncNet/data/models/DSN_ShapeNet_Table.pth)

### ComplementMe

[Airplane](https://shapenet.cs.stanford.edu/media/minhyuk/DeformSyncNet/data/models/DSN_ComplementMe_Airplane.pth), [Car](https://shapenet.cs.stanford.edu/media/minhyuk/DeformSyncNet/data/models/DSN_ComplementMe_Car.pth), [Chair](https://shapenet.cs.stanford.edu/media/minhyuk/DeformSyncNet/data/models/DSN_ComplementMe_Chair.pth), [Sofa](https://shapenet.cs.stanford.edu/media/minhyuk/DeformSyncNet/data/models/DSN_ComplementMe_Sofa.pth), [Table](https://shapenet.cs.stanford.edu/media/minhyuk/DeformSyncNet/data/models/DSN_ComplementMe_Table.pth)

