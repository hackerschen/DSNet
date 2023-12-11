import argparse
import numpy as np

import os
import sys
import logging
import functools
from termcolor import colored


@functools.lru_cache()
def create_logger(output_dir, dist_rank=0, name=''):
    # create logger
    logger = logging.getLogger(name)
    logger.setLevel(logging.DEBUG)
    logger.propagate = False

    # create formatter
    fmt = '[%(asctime)s %(name)s] (%(filename)s %(lineno)d): %(levelname)s %(message)s'
    color_fmt = colored('[%(asctime)s %(name)s]', 'green') + \
                colored('(%(filename)s %(lineno)d)', 'yellow') + ': %(levelname)s %(message)s'

    # create console handlers for master process
    if dist_rank == 0:
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setLevel(logging.DEBUG)
        console_handler.setFormatter(
            logging.Formatter(fmt=color_fmt, datefmt='%Y-%m-%d %H:%M:%S'))
        logger.addHandler(console_handler)

    # create file handlers
    file_handler = logging.FileHandler(os.path.join(output_dir, f'log_rank{dist_rank}.txt'), mode='a')
    file_handler.setLevel(logging.DEBUG)
    file_handler.setFormatter(logging.Formatter(fmt=fmt, datefmt='%Y-%m-%d %H:%M:%S'))
    logger.addHandler(file_handler)

    return logger

class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def str2bool(v):
    if v.lower() in ['true', 1]:
        return True
    elif v.lower() in ['false', 0]:
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')

def cal_cos_simility(input):
    # input的shape为C, H, W
    C, H, W = input.shape
    result = 0.0
    for i in range(C):
        for j in range(C):
            xj = input[j]
            xi = input[i]
            result += (1 - np_cos_simility(xi, xj))/2.0
    result = result / (C**2)
    return result

def np_cos_simility(vec1, vec2):
    vec1 = vec1.flatten()
    vec2 = vec2.flatten()
    s = (np.linalg.norm(vec1) * np.linalg.norm(vec2))
    if s == 0:
        return 0
    return vec1.dot(vec2)/s

if __name__ == "__main__":
    a = np.ones(shape=(64, 256, 256))
    print(cal_cos_simility(a))

