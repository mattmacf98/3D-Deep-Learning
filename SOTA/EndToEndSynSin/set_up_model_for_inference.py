import torch
import torch.nn as nn
import sys
sys.path.insert(0, './synsin')

import os
os.environ['DEBUG'] = '0'

from synsin.models.networks.sync_batchnorm import convert_model
from synsin.models.base_model import BaseModel
from synsin.options.options import get_model

def synsin_model(model_path):
    torch.backends.cudnn.enabled = True
    opts = torch.load(model_path)['opts']
    opts.render_ids = [1]
    torch_devices = [int(gpu_id.strip()) for gpu_id in opts.gpu_ids.split(',')]

    device = 'cuda:' + str(torch_devices[0])
    model = get_model(opts)
    if 'sync' in opts.norm_G:
        model = convert_model(model)
    model = nn.DataParallel(model, torch_devices[0:1]).cuda()

    model_to_test = BaseModel(model, opts)
    model_to_test.load_state_dict(torch.load(model_path)['state_dict'])
    model_to_test.eval()
    print("Loaded Model")
