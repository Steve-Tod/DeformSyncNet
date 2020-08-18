import logging
logger = logging.getLogger('base')

def define_net(opt_net):
    which_model = opt_net['model_type']

    if which_model == 'DNV0':
        from .DeformationNet import DeformationNetV0 as m
    elif which_model == 'DDNV1':
        from .DictNet import DeformationDictNetV1 as m
    elif which_model == 'RDDNV0':
        from .RawDictNet import RawDeformationDictNetV0 as m
    elif which_model == 'RDDNV1':
        from .RawDictNet import RawDeformationDictNetV1 as m
    elif which_model == '3DN':
        from .ThreeDN import Deform3DN as m
    elif which_model == '3DNProj':
        from .ThreeDN import Deform3DNProjection as m
    elif which_model == 'CC':
        from .CycleConsistentDeformation import AE_Meta_AtlasNet as m
        
    else:
        raise NotImplementedError(
            'Model [{:s}] not recognized'.format(which_model))
        
    net = m(opt_net)
    logger.info('Model [%s] created' % which_model)
    if opt_net['init'] == 'xavier':
        from .init import xavier_init
        net.apply(xavier_init)
        logger.info('Model init with xavier_init')

    return net
