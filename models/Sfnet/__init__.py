"""
Network Initializations
"""

import logging
import importlib
import torch

from models.Sfnet.sfnet_dfnet import AlignedDFnetv2
from models.Sfnet.sfnet_resnet import DeepR50_SF_deeply, DeepR18_SF_deeply


def get_sf_df(criterion, classes=1):
    """
    Get Network Architecture based on arguments provided
    """
    net = AlignedDFnetv2(num_classes=classes, criterion=criterion)
    return net

def get_sf_res18(criterion, classes=1):
    """
    Get Network Architecture based on arguments provided
    """
    net = DeepR18_SF_deeply(num_classes=classes, criterion=criterion)
    return net

def get_sf_res50(criterion, classes=1):
    """
    Get Network Architecture based on arguments provided
    """
    net = DeepR50_SF_deeply(num_classes=classes, criterion=criterion)
    return net

def get_net(args, criterion):
    """
    Get Network Architecture based on arguments provided
    """
    net = get_model(network=args.arch, num_classes=args.dataset_cls.num_classes,
                    criterion=criterion)
    num_params = sum([param.nelement() for param in net.parameters()])
    logging.info('Model params = {:2.1f}M'.format(num_params / 1000000))

    net = net.cuda()
    return net


def wrap_network_in_dataparallel(net, use_apex_data_parallel=False):
    """
    Wrap the network in Dataparallel
    """
    if use_apex_data_parallel:
        import apex
        net = apex.parallel.DistributedDataParallel(net)
    else:
        net = torch.nn.DataParallel(net)
    return net


def get_model(network, num_classes, criterion):
    """
    Fetch Network Function Pointer
    """
    module = network[:network.rfind('.')]
    model = network[network.rfind('.') + 1:]
    mod = importlib.import_module(module)
    net_func = getattr(mod, model)
    net = net_func(num_classes=num_classes, criterion=criterion)
    return net

