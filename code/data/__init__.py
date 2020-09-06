import torch
import torch.utils.data
import logging

logger = logging.getLogger('base')

def create_dataloader(dataset, dataset_opt, sampler=None):
    phase = dataset_opt['phase']
    if phase == 'train':
        num_workers = dataset_opt['num_worker']
        batch_size = dataset_opt['batch_size']
        shuffle = True
        return torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=shuffle,
                                           num_workers=num_workers, sampler=sampler, drop_last=True,
                                           pin_memory=True)
    else:
        return torch.utils.data.DataLoader(dataset, batch_size=1, shuffle=False, num_workers=4,
                                           pin_memory=True)

def create_dataset(dataset_opt):
    mode = dataset_opt['mode']
    logger = logging.getLogger('base')
    if mode == 'param_pair_v0':
        from .ParametricShape import ParametricPairTrainV0 as d
    elif mode == 'param_pair_v1':
        from .ParametricShape import ParametricPairTrainV1 as d
    elif mode == 'raw_pair_v0':
        from .RawShape import RawStructureNetPairTrainV0 as d
    elif mode == 'raw_pair_shapenet_v0':
        from .RawShape import RawShapeNetPairTrainV0 as d
    else:
        raise NotImplementedError('Dataset [{:s}] is not recognized.'.format(mode))
        
    dataset = d(dataset_opt)
    logger.info('Dataset [{:s} - {:s}] is created.'.format(dataset.__class__.__name__,
                                                           dataset_opt['name']))
    return dataset
