# 数据生成状态

## 📊 当前状态

**状态**: 🟢 **正在运行中**

- **进程**: 后台运行
- **当前进度**: 处理第一个目录（小说_1），约 7% (182/2500)
- **处理速度**: ~1-3 本小说/秒
- **预计剩余时间**: 约 1-2 小时（4 个目录）

---

## 🔍 监控命令

### 查看实时进度

```bash
# 查看日志尾部
tail -f /root/data/AI/dlm/jit/build_pretrain_data.log

# 查看进程状态
ps aux | grep build_pretrain_data.py | grep -v grep

# 查看已生成的文件数量
find /root/data/AI/dlm/jit/data/train /root/data/AI/dlm/jit/data/val -name "*.json" 2>/dev/null | wc -l
```

### 检查数据质量

```bash
# 查看统计信息（生成完成后）
cat /root/data/AI/dlm/jit/data/stats.json

# 检查训练集文件
ls -lh /root/data/AI/dlm/jit/data/train/ | head -10

# 检查验证集文件
ls -lh /root/data/AI/dlm/jit/data/val/ | head -10
```

---

## ⚠️ 注意事项

1. **文件生成时机**: 数据文件会在处理完所有小说后统一保存，所以处理过程中可能看不到文件
2. **处理速度**: 不同小说的长度差异很大，处理速度会波动
3. **错误处理**: 脚本会自动跳过无法处理的文件，继续处理其他文件
4. **内存使用**: 脚本会收集所有样本到内存，最后统一保存，注意内存使用情况

---

## 📝 配置信息

- **图像尺寸**: 64×64 = 4096 tokens
- **Token 范围**: 256-4096 tokens/样本（随机）
- **每个目录**: 2500 本小说
- **总目录数**: 4 个（小说_1, 小说_2, 小说_3, 小说_4）
- **数据集划分**: 99% 训练集，1% 验证集

---

**最后更新**: 2024-12-15
**状态**: 运行中
