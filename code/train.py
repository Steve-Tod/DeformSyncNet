import os
import math
import argparse
import random
import logging
from collections import OrderedDict

import torch
import matplotlib.pyplot as plt
import numpy as np

from option.parse import parse, dict2str, save_opt
from util import util_dir, util_log, util_misc
from data import create_dataloader, create_dataset
from solver import create_solver


def main():
    #### options
    parser = argparse.ArgumentParser()
    parser.add_argument('-opt', type=str, help='Path to option yaml file.')
    args = parser.parse_args()
    opt = parse(args.opt, is_train=True)

    #### loading resume state if exists
    if opt['path']['resume_state']:
        resume_state = torch.load(opt['path']['resume_state'])
        resume_path = opt['path']['resume_state']
        resume_step = os.path.basename(resume_path).split('.')[0]
        opt['path']['pretrain_model'] = os.path.join(opt['path']['model'], resume_step + '_model.pth')
        
    else:
        resume_state = None

    #### mkdir and loggers
    if resume_state is None:
        util_dir.mkdir_and_rename(
            opt['path']['experiment_root'], opt['time_stamp'])  # rename experiment folder if exists
        util_dir.mkdirs((path for key, path in opt['path'].items() if not key == 'experiment_root'
                         and 'pretrain_model' not in key 
                         and 'resume' not in key))
    # config loggers. Before it, the log will not work
    util_log.setup_logger('base', opt['path']['log'], 'train_' + opt['name'], level=logging.INFO,
                      screen=True, tofile=True)
    util_log.setup_logger('val', opt['path']['log'], 'val_' + opt['name'], level=logging.INFO,
                      screen=True, tofile=True)
    logger = logging.getLogger('base')
    val_logger = logging.getLogger('val')
    logger.info(dict2str(opt))
    
    # tensorboard logger
    if opt['use_tb_logger'] and 'debug' not in opt['name']:
        version = float(torch.__version__[0:3])
        from tensorboardX import SummaryWriter
        opt['path']['tb_logger'] = os.path.join(opt['path']['root'], 'tb_logger', opt['name'])
        if resume_state is None:
            util_dir.mkdir_and_rename(opt['path']['tb_logger'], opt['time_stamp'])
        tb_logger = SummaryWriter(log_dir=opt['path']['tb_logger'])

    #### random seed
    seed = opt['train']['manual_seed']
    if seed is None:
        seed = random.randint(1, 10000)
    logger.info('Random seed: {}'.format(seed))
    util_misc.set_random_seed(seed)


    #### create train and val dataloader
    for phase, dataset_opt in opt['dataset'].items():
        if phase == 'train':
            train_set = create_dataset(dataset_opt)
            train_loader = create_dataloader(train_set, dataset_opt)
            train_size = len(train_loader)
            total_iters = int(opt['train']['niter'])
            total_epochs = int(math.ceil(total_iters / train_size))
            logger.info('Number of train samples: {:,d}, iters: {:,d}'.format(len(train_set), train_size))
            logger.info('Total epochs needed: {:d} for iters {:,d}'.format(
                total_epochs, total_iters))
        elif phase == 'val':
            val_set = create_dataset(dataset_opt)
            val_loader = create_dataloader(val_set, dataset_opt)
            logger.info('Number of val samples in [{:s}]: {:d}'.format(
                dataset_opt['name'], len(val_set)))
        else:
            raise NotImplementedError('Phase [{:s}] is not recognized.'.format(phase))
    assert train_loader is not None

    if resume_state is None:
        save_opt(os.path.join(opt['path']['experiment_root'], 'opt.yaml'), opt)

    #### create solver
    solver = create_solver(opt)

    #### resume training
    if resume_state:
        logger.info('Resuming training from epoch: {}, iter: {}.'.format(
            resume_state['epoch'], resume_state['iter']))

        start_epoch = resume_state['epoch']
        solver.step = resume_state['iter']
        solver.resume_training(resume_state)  # handle optimizers and schedulers
    else:
        solver.step = 0
        start_epoch = 0

    #### training
    logger.info('Start training from epoch: {:d}, iter: {:d}'.format(start_epoch, solver.step))
    for epoch in range(start_epoch, total_epochs + 1):
        for _, train_data in enumerate(train_loader):
            if solver.step > total_iters:
                break
            #### update learning rate
            solver.update_learning_rate()

            #### training
            solver.feed_data(train_data)
            solver.optimize_parameter()

            #### log
            if solver.step % opt['logger']['print_freq'] == 0:
                if opt['use_tb_logger'] and 'debug' not in opt['name']:
                    solver.log_current(epoch, tb_logger)
                else:
                    solver.log_current(epoch)

            # validation
            if solver.step % opt['train']['val_freq'] == 0:
                val_logger.info('Start validation. Step: %d' % solver.step)
                idx = 0
                new_flag = True
                for val_data in val_loader:
                    solver.feed_data(val_data, True)
                    solver.test()
                    val_log = solver.evaluate()
                    if new_flag:
                        val_metric = OrderedDict()
                        for k, v in val_log.items():
                            val_metric[k] = [v]
                        new_flag = False
                    else:
                        for k, v in val_log.items():
                            val_metric[k].append(v)
                        
                    if idx < opt['logger']['num_save_image']:
                        fig = solver.get_current_visual()
                        idx += 1
                        save_path = os.path.join(opt['path']['visual'], '%d' % idx)
                        util_dir.mkdir(save_path)
                        fig.savefig(os.path.join(save_path, '%06d.png' % solver.step), dpi=200)
                        # tensorboard logger
                        if opt['use_tb_logger'] and 'debug' not in opt['name']:
                            tb_logger.add_figure('%d' % idx, fig, global_step=solver.step, close=False)
                        plt.close(fig)
                        
                # calc metric and log
                message_list = ['Step: %d ' % (solver.step)]
                for k, v in val_metric.items():
                    tmp_mean = np.mean(v)
                    tb_logger.add_scalar('val/' + k, tmp_mean, solver.step)
                    message_list.append('%s: %.4e' % (k, tmp_mean))
                if np.mean(val_metric[opt['train']['val_metric']]) < solver.best_res:
                    solver.best_step = solver.step
                    solver.best_res = np.mean(val_metric[opt['train']['val_metric']])
                    solver.save('best')
                message_list.append('Best %s: %.4e, step: %d' % (opt['train']['val_metric'], solver.best_res, solver.best_step))
                val_logger.info(' || '.join(message_list))
                        
            #### save models and training states
            if solver.step % opt['train']['save_freq'] == 0:
                logger.info('Saving models and training states.')
                solver.save(solver.step)
                solver.save_training_state(epoch)

    logger.info('Saving the final model.')
    solver.save('latest')
    logger.info('End of training.')
    
if __name__ == '__main__':
    main()
