# 预训练数据生成说明

## 🎯 数据生成配置

### 源数据
- **目录**: `/root/data/AI/creator/data/raw`
- **子目录**: `小说_1`, `小说_2`, `小说_3`, `小说_4`
- **每个目录采样**: 2500 本小说

### 数据分段策略

- **图像尺寸**: 64×64 = 4096 tokens
- **Token 长度范围**: 256-4096 tokens（随机）
- **分段方式**: 按照随机 token 长度分段，模拟真实 LLM 训练数据的多样性

### 数据集划分

- **训练集**: 99%
- **验证集**: 1%

---

## 📊 数据格式

### 样本格式

```json
{
  "text": "文本内容..."
}
```

### 输出目录结构

```
data/
├── train/          # 训练集（99%）
│   ├── train_0000.json
│   ├── train_0001.json
│   └── ...
├── val/            # 验证集（1%）
│   ├── val_0000.json
│   └── ...
└── stats.json      # 统计信息
```

### 统计信息格式

```json
{
  "total_samples": 48073,
  "train_samples": 47593,
  "val_samples": 480,
  "max_tokens": 4096,
  "min_tokens": 256,
  "img_size": 64,
  "samples_per_dir": 2500,
  "source_dirs": [...]
}
```

---

## 🚀 使用方法

### 生成数据

```bash
cd /root/data/AI/dlm/jit
source $(conda info --base)/etc/profile.d/conda.sh
conda activate seeme

# 后台运行（推荐）
nohup python build_pretrain_data.py > build_pretrain_data.log 2>&1 &

# 或者前台运行
python build_pretrain_data.py
```

### 检查进度

```bash
# 查看实时日志
tail -f build_pretrain_data.log

# 查看进程状态
ps aux | grep build_pretrain_data.py | grep -v grep

# 检查生成的文件数量
find data/train data/val -name "*.json" 2>/dev/null | wc -l

# 查看已生成的文件
ls -lh data/train/ | head -10
ls -lh data/val/ | head -10

# 查看统计信息（生成完成后）
cat data/stats.json
```

### 预计时间

- **处理速度**: ~1-2 本小说/秒
- **每个目录**: 2500 本小说，约 20-40 分钟
- **总时间**: 4 个目录，约 1.5-3 小时
- **文件保存**: 处理完所有小说后统一保存

---

## 📈 预期结果

### 数据量估算

- **每个目录**: 2500 本小说
- **总小说数**: 10,000 本
- **每本小说平均分段数**: ~5-10 段（取决于小说长度）
- **总样本数**: ~50,000-100,000 个样本

### Token 长度分布

- **最小**: 256 tokens
- **最大**: 4096 tokens
- **分布**: 随机（模拟真实 LLM 训练数据）

---

## ✅ 验证数据

### 检查数据格式

```python
import json

# 检查训练集
with open('data/train/train_0000.json', 'r', encoding='utf-8') as f:
    train_data = json.load(f)
    print(f"训练集样本数: {len(train_data)}")
    print(f"第一个样本: {train_data[0]['text'][:100]}...")

# 检查验证集
with open('data/val/val_0000.json', 'r', encoding='utf-8') as f:
    val_data = json.load(f)
    print(f"验证集样本数: {len(val_data)}")
```

### 使用数据集

```python
from dataset import TokenImageDataset
from transformers import AutoTokenizer

tokenizer = AutoTokenizer.from_pretrained('/root/data/AI/pretrain/Qwen2.5-7B-Instruct', trust_remote_code=True)

# 加载训练集
train_dataset = TokenImageDataset(
    data_path='./data/train',
    tokenizer=tokenizer,
    img_size=64,
)

# 加载验证集
val_dataset = TokenImageDataset(
    data_path='./data/val',
    tokenizer=tokenizer,
    img_size=64,
)

print(f"训练集: {len(train_dataset)} 个样本")
print(f"验证集: {len(val_dataset)} 个样本")
```

---

## 🔍 注意事项

1. **Token 长度**: 每个样本的 token 长度是随机的（256-4096），使用 padding mask 标记有效区域
2. **文件大小**: 每个 JSON 文件包含约 10,000 个样本，避免单个文件过大
3. **编码问题**: 脚本会自动尝试多种编码（utf-8, gbk, gb2312, gb18030）来读取文件
4. **处理时间**: 处理 10,000 本小说可能需要较长时间，建议在后台运行

---

## 📝 日志文件

数据生成过程的日志保存在 `build_pretrain_data.log`，包含：
- 处理进度
- 每个目录的样本数
- 错误和警告信息
- 最终统计信息

---

**生成日期**: 2024-12-15
**图像尺寸**: 64×64
**Token 范围**: 256-4096 tokens
