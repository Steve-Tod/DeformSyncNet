# Description of Train Options

Let us take [`train_RDDN_ShapeNet_airplane.yaml`](./train_RDDN_ShapeNet_airplane.yaml) as an example.

```yaml
name: RDDN_PointNetSeg_CD_shapenet_airplane_normcol_normpoint_c512 # experiment name
use_tb_logger: true # whether to use tensorboardX logger
gpu_id: [0] # GPU ID to use
type: RDNV1 # solver type, see more in code/solver/__init__.py


dataset:
  train:
    name: Raw ShapeNet Airplane Train # train set name
    mode: raw_pair_shapenet_v0 # train dataset type, see more in code/data/__init__.py
    root: /sailhome/jiangthu/jzy/projects/CycleConsistentDeformation/data/dataset_shapenet/ # path to full data
    category: Airplane # data category
    num_point: 2048 # number of points in sampled point cloud
    phase: train # pahse: train | val | test
    num_worker: 12 # number of threads for data loading
    batch_size: 32 # input batch size
    norm: True # whether to normalize input point cloud
  val:
    name: Raw ShapeNet Airplane Test # val set name
    mode: raw_pair_shapenet_v0 # val dataset type, see more in code/data/__init__.py
    root: /sailhome/jiangthu/jzy/projects/CycleConsistentDeformation/data/dataset_shapenet/ # path to full data
    category: Airplane # data category
    num_point: 2048 # number of points in sampled point cloud
    phase: val # pahse: train | val | test
    norm: True # whether to normalize input point cloud

model:
  init: xavier # network parameter initialization method, currently only xavier is supported
  model_type: RDDNV0 # network model type, see more in code/model/__init__.py
  dict: # dictionary net options
    version: 0 # dict net version, currently only 0 is supported
    arch: PointNetSeg # dict net architecture: MLP | PointNetSeg
    feature_dim: 1536 # dict net output column space dimension, should be 3X of coeff net out dimension
    norm_column: True # whether to normalize dict columns
  coeff: # coefficient net options
    version: 0 # coeff net version, 0(output difference of dst and src coeff) | 1(compute coeff from the concatenation of src and dst feature)
    arch: PointNetCls # coeff net architecture: PointNetCls for v0, PointNetMix for v1 
    out_dim: 512 # coeff net output dimention

train:
  learning_rate: 0.0005 # learning rate
  loss: # losses
    fit_CD: # fitting loss
      loss_type: CD # loss type: CD | EMD | l2
      weight: 100 # loss weight
    sym_CD: # symmetry loss
      loss_type: CD # loss type: CD | EMD | l2
      weight: 100 # loss weight
      sym_axis: 2 # symmetry axis
      
  lr_gamma: 0.5 # multiplicative factor of learning rate decay
  lr_scheme: MultiStepLR # lr scheduler scheme, currently only MultiStepLR is supprted
  lr_steps: [4.0e4, 8.0e4, 1.2e5] # lr decay milestones
  niter: 1.5e5 # number of total iterations
  save_freq: 5.0e3 # model saving frequency
  val_freq: 5.0e3 # validation frequency
  val_metric: loss_fit_CD # metric for best model
  weight_decay: 0 # weight decay

logger:
  print_freq: 500 # log frequency
  num_save_image: 10 # number of images to save during validation

path:
  root: ../ # project root

```

