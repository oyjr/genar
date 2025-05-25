# 🚀 MFBP项目数据结构迁移与代码重构指南

## 📋 项目概述

这是一个基于PyTorch Lightning的空间转录组学(ST)项目，使用MFBP模型进行基因表达预测。现需要将数据结构从原始HEST格式完全迁移到新的hest1k_datasets格式，并简化模型架构。

**核心变更**: 从双任务学习(基因表达+密度预测)简化为单任务学习(仅基因表达预测)，同时适配新的数据组织结构。

---

## 🔄 数据结构变更对比

### 原始数据结构 ❌
```
/data/ouyangjiarui/hest/processhest/
├── adata/                          # ST数据 (.h5ad)
│   ├── TENX115.h5ad
│   └── TENX117.h5ad
├── emb/                           # 嵌入特征 (.h5)
│   └── global/uni_v1/
│       ├── TENX115.h5
│       └── TENX117.h5
├── patches/                       # 图像patches (.h5)
│   ├── TENX115.h5
│   └── TENX117.h5
└── splits/                        # 数据划分 (.csv, .json)
    └── SKCM/
        ├── train_0.csv            # 训练集样本ID
        ├── test_0.csv             # 测试集样本ID
        ├── mean_50genes.json      # 目标基因列表
        ├── mean_200genes.json     # 密度基因列表
        ├── normalized_weights.json # 密度权重
        └── ids.csv                # 疾病样本ID
```

### 新数据结构 ✅
```
/data/ouyangjiarui/stem/hest1k_datasets/
├── PRAD/                          # 数据集名称
│   ├── wsis/                      # 全切片图像 (WSI)
│   │   ├── slide1.svs
│   │   └── slide2.svs
│   ├── st/                        # ST h5ad数据 (格式不变)
│   │   ├── MEND139.h5ad
│   │   ├── MEND140.h5ad
│   │   └── ...
│   └── processed_data/            # 处理后的数据
│       ├── 1spot_uni_ebd/         # UNI嵌入 (.pt文件)
│       │   ├── MEND139_uni.pt     # [spots, 1024]
│       │   ├── MEND140_uni.pt
│       │   └── ...
│       ├── 1spot_uni_ebd_aug/     # UNI增强嵌入
│       ├── 1spot_conch_ebd/       # CONCH嵌入
│       ├── 1spot_conch_ebd_aug/   # CONCH增强嵌入
│       ├── all_slide_lst.txt      # 所有slide ID列表 (23个)
│       └── selected_gene_list.txt # 选定基因列表 (200个)
├── her2st/                        # 其他数据集
├── kidney/
└── mouse_brain/
```

---

## 📊 关键文件格式确认

| 文件类型 | 格式 | 内容描述 | 示例 | 加载方式 |
|---------|------|----------|------|----------|
| **嵌入文件** | `.pt` | torch.Tensor, 形状`[spots, 1024]` | `MEND139_uni.pt` | `torch.load(file, weights_only=True)` |
| **基因列表** | `.txt` | 每行一个基因名，共200个基因 | `A2M\nACTB\nACTG1...` | `open().readlines()` |
| **Slide列表** | `.txt` | 每行一个slide ID，共23个slides | `SPA154\nSPA153\nSPA152...` | `open().readlines()` |
| **ST数据** | `.h5ad` | AnnData格式，保持不变 | 原有格式 | `scanpy.read_h5ad()` |

### 文件命名规范
- **嵌入文件**: `{slide_id}_{encoder_name}.pt` (如: `MEND139_uni.pt`, `SPA154_conch.pt`)
- **ST文件**: `{slide_id}.h5ad` (如: `MEND139.h5ad`)
- **编码器类型**: `uni` 或 `conch`
- **增强标识**: 目录名包含`_aug`后缀

---

## 🎯 新命令行接口设计

### 目标命令格式
```bash
python src/main.py \
    --config config/hest/base_config.yaml \
    --expr_name PRAD \
    --data_path /data/ouyangjiarui/stem/hest1k_datasets/PRAD/ \
    --slide_val "SPA154,SPA153" \
    --slide_test "SPA152,SPA151" \
    --encoder_name uni \
    --use_augmented \
    --mode train
```

### 参数变更说明

#### 新增参数 ➕
- `--expr_name` (str, required): 数据集名称 (PRAD, her2st, kidney, mouse_brain)
- `--data_path` (str, required): 数据集根目录路径
- `--slide_val` (str, optional): 验证集slide ID，逗号分隔，默认为空
- `--slide_test` (str, optional): 测试集slide ID，逗号分隔，默认为空
- `--encoder_name` (str, default='uni'): 编码器类型，choices=['uni', 'conch']
- `--use_augmented` (flag): 是否使用增强嵌入，默认False

#### 移除参数 ➖
- `--disease_type`: 由`--expr_name`替代
- `--fold`: 由slide-based划分替代

#### 保留参数 ✅
- `--config`: 配置文件路径
- `--mode`: 运行模式 (train/test)

---

## 🔧 核心修改任务清单

### 1. 主程序修改 (`src/main.py`)

#### 任务1.1: 更新参数解析器
- [ ] **添加新参数**: 在`get_parse()`函数中添加所有新参数
- [ ] **移除旧参数**: 删除`--disease_type`和`--fold`参数
- [ ] **参数验证**: 添加路径存在性检查和encoder_name有效性验证
- [ ] **默认值设置**: 为可选参数设置合理默认值

#### 任务1.2: 更新配置传递
- [ ] **配置对象更新**: 将新参数添加到config对象
- [ ] **移除旧配置**: 删除disease_type和fold相关配置
- [ ] **路径规范化**: 确保data_path以'/'结尾
- [ ] **slide列表解析**: 处理逗号分隔的slide ID字符串

### 2. 数据集类重构 (`src/dataset/hest_dataset.py`)

#### 任务2.1: 完全移除BKDDataset类
- [ ] **删除类定义**: 移除整个BKDDataset类 (约300行代码)
- [ ] **移除density相关**: 删除所有density_genes, density_adata_dict等
- [ ] **移除权重加载**: 删除normalized_weights.json相关代码
- [ ] **清理导入**: 移除BKDDataset相关的导入语句

#### 任务2.2: 重构STDataset类构造函数
```python
def __init__(self,
             mode: str,                    # 'train', 'val', 'test'
             data_path: str,               # 数据集根路径
             expr_name: str,               # 数据集名称
             slide_val: str = '',          # 验证集slides
             slide_test: str = '',         # 测试集slides
             encoder_name: str = 'uni',    # 编码器类型
             use_augmented: bool = False,  # 是否使用增强
             normalize: bool = True,       # 数据归一化
             cpm: bool = True,            # CPM归一化
             smooth: bool = True):        # 高斯平滑
```

#### 任务2.3: 路径构建逻辑重写
- [ ] **ST数据路径**: `self.st_dir = f"{data_path}/st"`
- [ ] **处理数据路径**: `self.processed_dir = f"{data_path}/processed_data"`
- [ ] **嵌入路径构建**: 根据encoder_name和use_augmented动态构建
- [ ] **文件存在性检查**: 验证关键目录和文件是否存在

#### 任务2.4: 关键方法实现

##### `load_gene_list()` 方法
- [ ] **文件读取**: 从`selected_gene_list.txt`读取基因列表
- [ ] **数据清理**: 去除空行和空白字符
- [ ] **错误处理**: 文件不存在或格式错误的处理
- [ ] **返回格式**: 返回基因名称列表

##### `load_slide_splits()` 方法
- [ ] **读取所有slides**: 从`all_slide_lst.txt`获取完整列表
- [ ] **解析输入**: 处理逗号分隔的slide ID字符串
- [ ] **去重和验证**: 确保slide ID有效且无重复
- [ ] **自动划分**: 剩余slides自动分配给训练集
- [ ] **返回字典**: `{'train': [...], 'val': [...], 'test': [...]}`

##### `load_emb()` 方法
- [ ] **文件路径构建**: `{emb_dir}/{slide_id}_{encoder_name}.pt`
- [ ] **PyTorch加载**: 使用`torch.load(weights_only=True)`
- [ ] **索引处理**: 支持单个spot或全部spots的加载
- [ ] **维度检查**: 确保返回正确的tensor维度
- [ ] **错误处理**: 文件不存在或格式错误的处理

#### 任务2.5: 数据加载逻辑更新

##### 训练模式 (`mode='train'`)
- [ ] **预加载ST数据**: 为所有训练slides预加载adata
- [ ] **计算累积长度**: 用于spot-level的索引映射
- [ ] **位置信息提取**: 从adata.obs获取array_row, array_col
- [ ] **数据增强**: 如果启用，应用相应的transforms

##### 验证/测试模式 (`mode='val'/'test'`)
- [ ] **按slide加载**: 每次返回完整slide的数据
- [ ] **延迟加载**: 避免内存占用过大
- [ ] **批处理支持**: 支持batch_size=1的处理

#### 任务2.6: `__getitem__()` 方法重写

##### 训练模式返回格式
```python
{
    'img': features,           # [1, 1024] 或 [1024]
    'target_genes': expression, # [num_genes]
    'positions': positions,     # [2] (array_row, array_col)
    'slide_id': slide_id,      # str
    'spot_idx': spot_idx       # int
}
```

##### 验证/测试模式返回格式
```python
{
    'img': features,           # [num_spots, 1024]
    'target_genes': expression, # [num_spots, num_genes]
    'positions': positions,     # [num_spots, 2]
    'slide_id': slide_id,      # str
    'num_spots': num_spots     # int
}
```

### 3. 数据接口更新 (`src/dataset/data_interface.py`)

#### 任务3.1: 简化DataInterface类
- [ ] **移除BKDDataset**: 统一使用STDataset
- [ ] **参数传递更新**: 传递新的参数到数据集类
- [ ] **setup方法简化**: 移除复杂的条件判断
- [ ] **错误处理**: 添加数据集创建失败的处理

#### 任务3.2: 参数映射
```python
base_params = {
    'data_path': self.config.data_path,
    'expr_name': self.config.expr_name,
    'slide_val': self.config.slide_val,
    'slide_test': self.config.slide_test,
    'encoder_name': self.config.encoder_name,
    'use_augmented': self.config.use_augmented,
    'normalize': self.config.DATA.normalize,
    'cpm': self.config.DATA.cpm,
    'smooth': self.config.DATA.smooth,
}
```

### 4. 模型代码简化 (`src/model/`)

#### 任务4.1: 移除密度预测功能

##### MFBP模型类 (`src/model/MFBP/MFBP.py`)
- [ ] **删除密度预测器**: 移除DensityPredictor相关代码
- [ ] **删除密度编码器**: 移除DensityEncoder相关代码
- [ ] **简化forward方法**: 只返回基因表达预测
- [ ] **移除密度损失**: 删除density_loss_weight和相关计算

##### 模型接口 (`src/model/model_interface.py`)
- [ ] **简化training_step**: 移除密度相关的损失计算
- [ ] **简化validation_step**: 只计算基因表达损失
- [ ] **更新指标计算**: 移除密度相关指标
- [ ] **简化_compute_loss**: 只计算MSE和相关损失

#### 任务4.2: 更新损失计算
- [ ] **单一损失函数**: 只使用MSE损失计算基因表达
- [ ] **移除权重**: 删除density_loss_weight相关逻辑
- [ ] **更新监控指标**: 从val_mse改回val_loss
- [ ] **简化日志记录**: 移除密度相关的日志

#### 任务4.3: 模型输出格式
```python
# 简化后的输出格式
outputs = {
    'logits': gene_predictions  # [batch_size, num_genes]
}
```

### 5. 配置文件更新 (`config/`)

#### 任务5.1: 基础配置文件 (`config/hest/base_config.yaml`)
- [ ] **移除密度配置**: 删除所有density相关配置
- [ ] **更新监控指标**: monitor从val_mse改为val_loss
- [ ] **简化数据配置**: 移除BKDDataset相关配置
- [ ] **更新日志路径**: 适配新的数据集结构

#### 任务5.2: 数据集特定配置
- [ ] **创建通用配置**: 不再需要疾病特定的配置文件
- [ ] **参数化设计**: 通过命令行参数而非配置文件指定数据集
- [ ] **保持向前兼容**: 确保现有的训练脚本仍可使用

### 6. 导入和初始化更新

#### 任务6.1: 数据集导入 (`src/dataset/__init__.py`)
- [ ] **移除BKDDataset**: 从__all__中删除
- [ ] **保留STDataset**: 确保正确导入
- [ ] **清理未使用导入**: 移除相关的导入语句

#### 任务6.2: 模型导入检查
- [ ] **验证模型导入**: 确保MFBP模型正确导入
- [ ] **清理未使用模块**: 移除密度相关的模块导入

---

## 🎯 实现优先级与详细步骤

### 🔥 优先级1 (核心功能) - 必须完成

#### 步骤1: 路径系统重构 (预计2小时)
1. **更新STDataset构造函数**
   - 修改参数列表
   - 重写路径构建逻辑
   - 添加基本的错误检查

2. **实现基础文件加载**
   - `load_gene_list()`方法
   - `load_slide_splits()`方法
   - 基本的文件存在性检查

#### 步骤2: 嵌入文件加载重构 (预计1.5小时)
1. **更新load_emb方法**
   - 从HDF5改为PyTorch格式
   - 处理文件路径构建
   - 添加错误处理

2. **测试嵌入加载**
   - 验证tensor形状
   - 测试索引功能
   - 确认数据类型

#### 步骤3: 数据划分逻辑 (预计1小时)
1. **实现slide划分**
   - 解析命令行参数
   - 验证slide ID有效性
   - 自动分配训练集

2. **更新数据集创建**
   - 修改__getitem__方法
   - 处理不同模式的数据返回
   - 测试数据加载

#### 步骤4: 命令行接口 (预计0.5小时)
1. **更新main.py**
   - 添加新参数
   - 移除旧参数
   - 更新配置传递

### ⚡ 优先级2 (功能简化) - 重要

#### 步骤5: 移除密度预测 (预计2小时)
1. **删除BKDDataset类**
   - 完全移除类定义
   - 清理相关导入
   - 更新__init__.py

2. **简化模型代码**
   - 移除密度预测器
   - 简化forward方法
   - 更新损失计算

#### 步骤6: 配置文件更新 (预计0.5小时)
1. **简化配置结构**
   - 移除密度相关配置
   - 更新监控指标
   - 清理未使用配置

### 🔧 优先级3 (完善功能) - 可选

#### 步骤7: 错误处理和验证 (预计1小时)
1. **添加全面的错误处理**
   - 文件不存在处理
   - 数据格式验证
   - 参数有效性检查

2. **添加日志和调试信息**
   - 数据加载进度
   - 错误详细信息
   - 性能监控

---

## ✅ 详细验证检查清单

### 数据加载验证
- [ ] **基因列表加载**
  - [ ] 能正确读取selected_gene_list.txt
  - [ ] 基因数量正确 (200个)
  - [ ] 基因名称格式正确
  - [ ] 处理空行和特殊字符

- [ ] **Slide列表加载**
  - [ ] 能正确读取all_slide_lst.txt
  - [ ] Slide数量正确 (23个)
  - [ ] Slide ID格式验证
  - [ ] 重复ID检测

- [ ] **嵌入文件加载**
  - [ ] 能正确加载.pt文件
  - [ ] Tensor形状正确 [spots, 1024]
  - [ ] 数据类型正确 (float32)
  - [ ] 索引功能正常

- [ ] **ST数据加载**
  - [ ] h5ad文件正常读取
  - [ ] 基因过滤正确
  - [ ] 位置信息提取正确
  - [ ] 数据归一化正常

### 功能验证
- [ ] **Slide划分逻辑**
  - [ ] 验证集划分正确
  - [ ] 测试集划分正确
  - [ ] 训练集自动分配
  - [ ] 无重复和遗漏

- [ ] **数据集创建**
  - [ ] 训练数据集创建成功
  - [ ] 验证数据集创建成功
  - [ ] 测试数据集创建成功
  - [ ] 数据加载器正常工作

- [ ] **模型接口**
  - [ ] 前向传播正常
  - [ ] 输出格式正确
  - [ ] 梯度计算正常
  - [ ] 内存使用合理

### 训练验证
- [ ] **训练启动**
  - [ ] 命令行参数解析正确
  - [ ] 配置加载成功
  - [ ] 数据模块初始化
  - [ ] 模型初始化成功

- [ ] **训练过程**
  - [ ] 损失正常下降
  - [ ] 验证指标计算
  - [ ] 学习率调度正常
  - [ ] 早停机制工作

- [ ] **模型保存**
  - [ ] 检查点正常保存
  - [ ] 最佳模型记录
  - [ ] 日志文件生成
  - [ ] 指标记录完整

### 性能验证
- [ ] **内存使用**
  - [ ] 训练时内存稳定
  - [ ] 无内存泄漏
  - [ ] 大数据集支持
  - [ ] GPU内存优化

- [ ] **速度性能**
  - [ ] 数据加载速度
  - [ ] 训练速度合理
  - [ ] 验证速度正常
  - [ ] 整体效率提升

---

## 🚨 重要注意事项与边界情况

### ⚠️ 完全迁移原则
- **零向后兼容**: 完全删除旧格式支持，不保留任何兼容代码
- **彻底清理**: 移除所有未使用的代码、配置和导入
- **统一接口**: 所有数据集使用相同的STDataset类和接口

### 🔍 关键变更点详解

#### 1. 文件格式变更
- **嵌入文件**: `.h5` (HDF5) → `.pt` (PyTorch)
- **基因列表**: `.json` → `.txt`
- **数据划分**: `.csv` → 命令行参数
- **权重文件**: 完全移除

#### 2. 路径结构变更
- **多级目录**: `emb/global/uni_v1/` → `processed_data/1spot_uni_ebd/`
- **扁平化设计**: 减少目录层级，简化路径构建
- **统一命名**: 所有文件遵循统一的命名规范

#### 3. 参数系统变更
- **Fold-based**: `--fold 0` → **Slide-based**: `--slide_val "id1,id2"`
- **疾病类型**: `--disease_type SKCM` → **数据集名称**: `--expr_name PRAD`
- **路径指定**: 从配置文件 → 命令行参数

#### 4. 模型功能变更
- **双任务**: 基因表达 + 密度预测 → **单任务**: 仅基因表达预测
- **复杂损失**: MSE + Density Loss → **简单损失**: 仅MSE
- **多输出**: logits + density_pred → **单输出**: 仅logits

### 📝 代码质量要求

#### 错误处理要求
- **文件检查**: 所有文件操作前检查存在性
- **格式验证**: 验证数据格式和内容正确性
- **异常捕获**: 使用try-except处理所有可能的异常
- **错误信息**: 提供清晰、有用的错误信息

#### 代码规范要求
- **注释完整**: 所有新方法和复杂逻辑添加注释
- **变量命名**: 使用描述性的变量名
- **类型提示**: 为函数参数和返回值添加类型提示
- **代码整洁**: 移除所有调试打印和临时代码

#### 性能要求
- **内存优化**: 避免不必要的数据复制和缓存
- **延迟加载**: 大文件使用延迟加载策略
- **批处理**: 支持高效的批处理操作
- **GPU利用**: 确保GPU内存使用合理

### 🔧 边界情况处理

#### 数据边界情况
- **空slide列表**: 验证或测试集为空的处理
- **重复slide ID**: 检测和处理重复的slide ID
- **缺失文件**: 某些slide缺少嵌入或ST文件
- **格式错误**: 文件格式不正确或损坏

#### 参数边界情况
- **无效路径**: 数据路径不存在或无权限
- **错误编码器**: 指定不支持的编码器类型
- **slide不存在**: 指定的slide ID在数据集中不存在
- **配置冲突**: 命令行参数与配置文件冲突

#### 运行时边界情况
- **内存不足**: 大数据集导致内存溢出
- **GPU内存**: GPU内存不足的处理
- **网络中断**: 分布式训练时的网络问题
- **磁盘空间**: 日志和检查点文件的磁盘空间

---

## 🎯 预期最终效果与测试方案

### 最终效果目标

#### 1. 功能完整性
- **数据加载**: 能够正确加载所有新格式的数据文件
- **模型训练**: 训练过程稳定，收敛正常
- **多数据集**: 支持PRAD、her2st、kidney、mouse_brain等数据集
- **灵活配置**: 通过命令行参数灵活配置训练参数

#### 2. 性能指标
- **训练速度**: 相比原版本，训练速度提升20%以上
- **内存使用**: 内存使用减少30%以上 (移除密度预测)
- **代码简洁**: 代码行数减少25%以上
- **维护性**: 代码结构更清晰，易于维护和扩展

#### 3. 用户体验
- **命令简洁**: 一行命令即可启动训练
- **错误友好**: 清晰的错误信息和解决建议
- **日志完整**: 详细的训练日志和进度信息
- **文档完善**: 完整的使用文档和示例

### 详细测试方案

#### 测试阶段1: 单元测试
```bash
# 测试数据加载
python -c "
from src.dataset.hest_dataset import STDataset
dataset = STDataset(
    mode='train',
    data_path='/data/ouyangjiarui/stem/hest1k_datasets/PRAD/',
    expr_name='PRAD',
    slide_val='SPA154,SPA153',
    slide_test='SPA152,SPA151',
    encoder_name='uni'
)
print(f'Dataset length: {len(dataset)}')
sample = dataset[0]
print(f'Sample keys: {sample.keys()}')
print(f'Feature shape: {sample[\"img\"].shape}')
print(f'Gene shape: {sample[\"target_genes\"].shape}')
"
```

#### 测试阶段2: 集成测试
```bash
# 测试完整训练流程
python src/main.py \
    --config config/hest/base_config.yaml \
    --expr_name PRAD \
    --data_path /data/ouyangjiarui/stem/hest1k_datasets/PRAD/ \
    --slide_val "SPA154,SPA153" \
    --slide_test "SPA152,SPA151" \
    --encoder_name uni \
    --mode train \
    --max_epochs 2  # 快速测试
```

#### 测试阶段3: 多数据集测试
```bash
# 测试不同数据集
for dataset in PRAD her2st kidney mouse_brain; do
    echo "Testing dataset: $dataset"
    python src/main.py \
        --config config/hest/base_config.yaml \
        --expr_name $dataset \
        --data_path /data/ouyangjiarui/stem/hest1k_datasets/$dataset/ \
        --encoder_name uni \
        --mode train \
        --max_epochs 1
done
```

#### 测试阶段4: 性能测试
```bash
# 内存和速度测试
python -m memory_profiler src/main.py \
    --config config/hest/base_config.yaml \
    --expr_name PRAD \
    --data_path /data/ouyangjiarui/stem/hest1k_datasets/PRAD/ \
    --encoder_name uni \
    --mode train \
    --max_epochs 5
```
