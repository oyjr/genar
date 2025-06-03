#!/usr/bin/env python
import sys
sys.path.insert(0, 'src')
from main import build_config_from_args

class Args:
    dataset = 'PRAD'
    model = 'VAR_ST'
    encoder = None
    gpus = 1
    epochs = 1
    batch_size = 2
    lr = None
    weight_decay = None
    patience = None
    strategy = 'auto'
    sync_batchnorm = False
    use_augmented = True
    expand_augmented = True
    mode = 'train'
    seed = None
    config = None

args = Args()
config = build_config_from_args(args)

print('=== CONFIG.MODEL 内容 ===')
for key, value in config.MODEL.items():
    print(f'{key}: {value}')

print(f'\n=== 关键参数检查 ===')
print(f'feature_dim: {getattr(config.MODEL, "feature_dim", "NOT FOUND")}')
print(f'histology_feature_dim: {getattr(config.MODEL, "histology_feature_dim", "NOT FOUND")}') 