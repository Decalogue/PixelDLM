import os
import json
from utils import ensure_dir


def create_test_texts(output_path: str = './data/test/texts.json'):
    """创建测试文本数据"""
    
    # 基础测试文本（不同长度和类型）
    base_texts = [
        # 短文本
        "你好",
        "什么是AI？",
        "Python编程",
        "机器学习",
        "深度学习",
        
        # 中等长度文本
        "什么是机器学习？",
        "机器学习是人工智能的一个分支。",
        "Python 是一种高级编程语言。",
        "深度学习使用神经网络来学习数据的表示。",
        "Transformer 模型在 NLP 领域取得了巨大成功。",
        
        # 长文本
        "机器学习是人工智能的一个分支，它使计算机能够从数据中学习，而无需明确编程。深度学习是机器学习的一个子集，使用多层神经网络来学习数据的复杂表示。",
        "Python 是一种高级编程语言，以其简洁的语法和强大的功能而闻名。它广泛用于数据科学、机器学习和 Web 开发。",
        "Transformer 模型是一种基于注意力机制的神经网络架构，在自然语言处理任务中取得了革命性的突破。",
        
        # 技术文本
        "注意力机制允许模型在处理序列时关注不同位置的信息，这使得 Transformer 能够更好地理解长距离依赖关系。",
        "自注意力机制是 Transformer 的核心组件，它计算序列中每个位置与其他所有位置之间的相关性。",
        "位置编码用于为 Transformer 模型提供序列中每个位置的相对或绝对位置信息。",
        
        # 代码相关
        "def train_model(model, data, epochs):\n    for epoch in range(epochs):\n        loss = model.train_step(data)\n        print(f'Epoch {epoch}, Loss: {loss}')",
        "import torch\nimport torch.nn as nn\n\nclass MLP(nn.Module):\n    def __init__(self, input_dim, hidden_dim, output_dim):\n        super().__init__()\n        self.fc1 = nn.Linear(input_dim, hidden_dim)\n        self.fc2 = nn.Linear(hidden_dim, output_dim)",
        
        # 中文文本
        "自然语言处理是计算机科学和人工智能的一个分支，研究如何让计算机理解和生成人类语言。",
        "大语言模型是近年来人工智能领域的重要突破，能够理解和生成高质量的自然语言文本。",
        "预训练语言模型通过在大规模文本数据上学习语言的统计规律，获得了强大的语言理解能力。",
        
        # 混合语言
        "Machine Learning (机器学习) 是 AI 的一个重要分支。",
        "深度学习 (Deep Learning) 使用多层神经网络来学习数据的表示。",

        # 问答对
        "问题：什么是梯度下降？答案：梯度下降是一种优化算法，用于最小化损失函数。",
        "问题：什么是反向传播？答案：反向传播是训练神经网络时计算梯度的方法。",
        "问题：什么是过拟合？答案：过拟合是指模型在训练数据上表现很好，但在测试数据上表现较差的现象。",
    ]
    
    # 生成更多测试文本
    test_texts = base_texts.copy()
    
    # 通过组合和变化生成更多文本
    prefixes = ["什么是", "如何", "为什么", "请解释", "介绍一下"]
    topics = ["机器学习", "深度学习", "神经网络", "自然语言处理", "计算机视觉", 
              "强化学习", "生成模型", "Transformer", "注意力机制"]
    
    for prefix in prefixes:
        for topic in topics:
            test_texts.append(f"{prefix}{topic}？")

    # 保存
    ensure_dir(os.path.dirname(output_path))
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(test_texts, f, ensure_ascii=False, indent=2)
    
    print(f"已创建 {len(test_texts)} 条测试文本")
    print(f"保存到: {output_path}")


if __name__ == '__main__':
    create_test_texts('./data/test/texts.json')
