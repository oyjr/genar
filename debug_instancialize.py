#!/usr/bin/env python
import sys
sys.path.insert(0, 'src')
import torch
import inspect
from main import build_config_from_args
from model.VAR.VAR_ST_Complete import VAR_ST_Complete
from addict import Dict as AddictDict

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

print("=== 🔧 调试 instancialize 方法 ===")

# Step 1: 构建配置
args = Args()
config = build_config_from_args(args)

print(f"\n=== Step 1: 配置检查 ===")
print(f"config.MODEL keys: {list(config.MODEL.keys())}")
for key, value in config.MODEL.items():
    print(f"  {key}: {value}")

# Step 2: 模拟 instancialize 方法
print(f"\n=== Step 2: 模拟 instancialize 方法 ===")

# 获取模型初始化参数
Model = VAR_ST_Complete
class_args = inspect.getfullargspec(Model.__init__).args[1:]
print(f"VAR_ST_Complete.__init__ 参数: {class_args}")

# 处理model_config
model_config = config.MODEL
print(f"\nmodel_config 类型: {type(model_config)}")
print(f"model_config 内容: {model_config}")

# 检查是否有 __dict__ 属性
if isinstance(model_config, AddictDict):
    print("model_config 是 addict.Dict，使用 dict() 转换")
    model_config_dict = dict(model_config)
    inkeys = model_config_dict.keys()
elif hasattr(model_config, '__dict__'):
    print("model_config 有 __dict__ 属性")
    model_config_dict = vars(model_config)
    inkeys = model_config_dict.keys()
else:
    print("model_config 没有 __dict__ 属性，当作字典处理")
    model_config_dict = model_config
    inkeys = model_config_dict.keys()

print(f"\nmodel_config_dict keys: {list(inkeys)}")
print(f"model_config_dict:")
for key, value in model_config_dict.items():
    print(f"  {key}: {value}")

args1 = {}

# 从配置中获取参数
print(f"\n=== 参数匹配过程 ===")
for arg in class_args:
    if arg in inkeys:
        args1[arg] = model_config_dict[arg]
        print(f"✅ 找到参数 {arg}: {model_config_dict[arg]}")
    elif arg == 'config':
        args1[arg] = config
        print(f"✅ 特殊参数 config: 传入完整配置")
    else:
        print(f"❌ 未找到参数 {arg}")

print(f"\n=== 最终传递给模型的参数 ===")
for key, value in args1.items():
    print(f"  {key}: {value}")

# Step 3: 尝试创建模型
print(f"\n=== Step 3: 尝试创建模型 ===")
try:
    model = Model(**args1)
    print(f"✅ 模型创建成功！")
    print(f"model.histology_feature_dim: {model.histology_feature_dim}")
except Exception as e:
    print(f"❌ 模型创建失败！")
    print(f"错误: {e}")
    import traceback
    traceback.print_exc() 