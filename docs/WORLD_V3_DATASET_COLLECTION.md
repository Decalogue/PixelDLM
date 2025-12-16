# World v3 数据集收集指南

## 📊 数据集概览

### World v3 数据集信息

- **总数据量**: 3.1T tokens
- **语言**: 100+ 种语言（包括中文、英文等）
- **领域**: Web、图书、代码、科学、小说、数学、法律等
- **用途**: 训练 RWKV-7 "Goose" World 模型系列

### 数据组成（参考）

| 类别 | Token 数 (B) | 占比 |
|------|-------------|------|
| 网络 (Web) | 1945.2 | 62.4% |
| 图书 (Books) | 337.2 | 10.8% |
| 代码 (Code) | 258.4 | 8.3% |
| 科学与维基 | 222.7 | 7.1% |
| 小说 (Fiction) | 192.6 | 6.2% |
| 聊天、问答与指令 | 110.0 | 3.5% |
| 数学 (Math) | 32.3 | 1.0% |
| 法律与政府 | 19.0 | 0.6% |
| 诗歌与歌词 | 1.7 | 0.1% |
| **总计** | **3119.2** | **100%** |

---

## 🔗 数据获取方式

### 方式 1：Hugging Face（推荐）⭐

#### 官方仓库

1. **Goose-World 组织**
   - 链接: https://huggingface.co/datasets/Goose-World/RWKV-World-v3
   - 提供详细的数据集信息和文档

2. **hevok 的集合**
   - 链接: https://huggingface.co/collections/hevok/rwkv-world-v3-corpus
   - 整合了 World v3 语料库的各个组件
   - **更容易访问和使用**

#### 下载方法

```bash
# 安装 huggingface_hub
pip install huggingface_hub

# 下载数据集（需要足够的存储空间）
from huggingface_hub import snapshot_download

# 下载整个数据集（需要大量存储空间）
snapshot_download(
    repo_id="Goose-World/RWKV-World-v3",
    repo_type="dataset",
    local_dir="./data/world_v3",
    local_dir_use_symlinks=False
)
```

**注意事项**：
- ⚠️ **数据量巨大**：3.1T tokens 需要大量存储空间
- ⚠️ **下载时间**：可能需要数天到数周（取决于网络速度）
- ⚠️ **带宽要求**：建议使用高速网络或服务器下载

---

### 方式 2：RWKV 官方文档

- **RWKV 中国文档**: https://rwkv.cn/docs/RWKV-Wiki/Dataset
- 提供数据集详细组成和来源信息
- 可能包含其他下载方式或镜像

---

## 📦 数据格式

### 预期格式

World v3 数据集通常以以下格式提供：

1. **Parquet 文件**（推荐）
   - 高效的列式存储格式
   - 适合大规模数据处理

2. **JSON Lines**（.jsonl）
   - 每行一个 JSON 对象
   - 易于流式处理

3. **文本文件**（.txt）
   - 纯文本格式
   - 可能需要预处理

### 数据字段

通常包含以下字段：
- `text`: 文本内容
- `metadata`: 元数据（来源、语言、领域等）
- `token_count`: Token 数量（如果已预处理）

---

## 💾 存储需求估算

### 原始数据大小

**假设**：
- 平均每个 token 约 4 字节（UTF-8 编码）
- 3.1T tokens ≈ 12.4TB 原始文本数据

**压缩后**：
- 使用 gzip/bzip2 压缩：约 3-5TB
- 使用 Parquet 格式：约 2-4TB

### 推荐存储配置

- **最小**: 5TB 可用空间
- **推荐**: 10TB+ 可用空间（考虑处理后的数据）
- **理想**: 20TB+ 可用空间（完整数据集 + 处理后的数据）

---

## 🚀 收集步骤

### 步骤 1：准备存储空间

```bash
# 检查可用空间
df -h

# 创建数据目录
mkdir -p /path/to/world_v3_data
```

### 步骤 2：安装依赖

```bash
pip install huggingface_hub datasets
```

### 步骤 3：下载数据

```python
from huggingface_hub import snapshot_download
from datasets import load_dataset

# 方式 A: 下载整个数据集（需要大量空间）
snapshot_download(
    repo_id="Goose-World/RWKV-World-v3",
    repo_type="dataset",
    local_dir="./data/world_v3",
    local_dir_use_symlinks=False
)

# 方式 B: 使用 datasets 库（支持流式加载）
dataset = load_dataset("Goose-World/RWKV-World-v3", streaming=True)
```

### 步骤 4：验证数据

```python
# 检查数据格式
for sample in dataset.take(1):
    print(sample.keys())
    print(sample['text'][:100])  # 查看前 100 个字符
```

---

## ⚠️ 收集难度评估

### 难度：中等 ⭐⭐⭐

**容易的部分**：
- ✅ 数据已公开在 Hugging Face
- ✅ 有官方文档和说明
- ✅ 可以使用标准工具下载

**困难的部分**：
- ⚠️ **数据量巨大**：3.1T tokens 需要大量存储
- ⚠️ **下载时间长**：可能需要数天到数周
- ⚠️ **网络要求高**：需要稳定的高速网络
- ⚠️ **存储成本**：需要大量存储空间

---

## 💡 替代方案

### 方案 1：使用 World v2.8（1T tokens）

**优势**：
- ✅ 数据量更小（1T vs 3.1T）
- ✅ 下载更快
- ✅ 存储需求更小（约 3-4TB）

**适用场景**：
- 快速测试和验证
- 资源有限的情况

### 方案 2：使用 World v2.9（2T tokens）

**优势**：
- ✅ 数据量适中（2T）
- ✅ 平衡了数据量和收集难度

### 方案 3：采样使用

**策略**：
- 下载完整数据集后，根据需求采样
- 例如：只使用中文数据、只使用小说数据等
- 可以大幅减少实际使用的数据量

---

## 📝 收集建议

### 1. 评估需求

**问题**：
- 是否需要完整的 3.1T tokens？
- 还是可以使用较小的版本（如 1T 或 2T）？
- 是否有足够的存储空间和带宽？

### 2. 分阶段收集

**策略**：
1. **第一阶段**：下载 100B-500B tokens 用于测试
2. **第二阶段**：根据测试结果决定是否下载更多
3. **第三阶段**：如果需要，下载完整数据集

### 3. 使用服务器下载

**建议**：
- 使用云服务器（如 AWS、GCP、Azure）
- 利用高速网络和存储
- 下载完成后传输到本地（如果需要）

### 4. 考虑成本

**成本因素**：
- 存储成本（云存储或本地存储）
- 网络传输成本
- 时间成本（下载和处理时间）

---

## ✅ 总结

### World v3 数据集收集

**可行性**：✅ **可行，但需要充分准备**

**关键点**：
1. ✅ 数据已公开在 Hugging Face
2. ⚠️ 需要大量存储空间（5-10TB+）
3. ⚠️ 需要长时间下载（数天到数周）
4. ⚠️ 需要稳定的高速网络

**推荐策略**：
- **快速测试**：使用 World v2.8（1T tokens）
- **完整训练**：使用 World v3（3.1T tokens）
- **资源有限**：考虑采样或使用部分数据

**收集难度**：⭐⭐⭐（中等）

---

**更新日期**: 2025-12-15
