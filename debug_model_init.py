#!/usr/bin/env python
import sys
sys.path.insert(0, 'src')
from main import build_config_from_args
from model import ModelInterface
import inspect

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

print('=== CONFIG.MODEL 检查 ===')
for key, value in config.MODEL.items():
    print(f'{key}: {value}')

# 模拟ModelInterface的初始化过程
print('\n=== 模拟 ModelInterface 初始化 ===')

# 检查VAR_ST_Complete的初始化参数
from model.VAR.VAR_ST_Complete import VAR_ST_Complete
class_args = inspect.getfullargspec(VAR_ST_Complete.__init__).args[1:]
print(f'VAR_ST_Complete.__init__ 参数: {class_args}')

# 模拟instancialize方法
model_config_dict = dict(config.MODEL)
inkeys = model_config_dict.keys()

args1 = {}
for arg in class_args:
    if arg in inkeys:
        args1[arg] = model_config_dict[arg]
        print(f'✅ 找到参数 {arg}: {model_config_dict[arg]}')
    else:
        print(f'❌ 未找到参数 {arg}')

print(f'\n=== 最终传递给模型的参数 ===')
for key, value in args1.items():
    print(f'{key}: {value}') 