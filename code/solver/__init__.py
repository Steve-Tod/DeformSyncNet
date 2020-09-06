import logging

logger = logging.getLogger('base')

def create_solver(opt):
    solver_type = opt['type']
    if solver_type == 'RDNV0':
        from .RawDeformationNetSolverV0 import RawDeformationNetSolverV0 as S
    elif solver_type == 'RDNV1': # with semantic segmentation results
        from .RawDeformationNetSolverV1 import RawDeformationNetSolverV1 as S
    elif solver_type == 'RDNV2': # use hard projections on deformed shapes
        from .RawDeformationNetSolverV2 import RawDeformationNetSolverV2 as S
    else:
        raise NotImplementedError(
            'Solver [%s] not recognized' % solver_type)
    solver = S(opt)
    logger.info('Solver [%s] created' % solver_type)
    return solver
