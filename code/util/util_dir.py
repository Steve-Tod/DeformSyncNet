import os
import logging
from datetime import datetime

def get_timestamp():
    return datetime.now().strftime('%y%m%d-%H%M%S')

def mkdir(path):
    if not os.path.exists(path):
        os.makedirs(path)


def mkdirs(paths):
    if isinstance(paths, str):
        mkdir(paths)
    else:
        for path in paths:
            mkdir(path)

def mkdir_and_rename(path, archive_name=None):
    if os.path.exists(path):
        new_name = path + '_archived_' + archive_name
        print('Path already exists. Rename it to [{:s}]'.format(new_name))
        logger = logging.getLogger('base')
        logger.info('Path already exists. Rename it to [{:s}]'.format(new_name))
        choice = input('Are you sure? y/[n]')
        if choice is not 'y':
            print('Give up renaming, exit')
            exit(0)
        os.rename(path, new_name)
    os.makedirs(path)
