"""
RNN文本生成模型 - 基于Time Machine数据集
单词级文本预测，使用原始RNN实现
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import numpy as np
import urllib.request
import os
import re
import random


# ==================== 超参数配置 ====================
SEQ_LEN = 32          # 序列长度
EMBEDDING_DIM = 128   # 词向量维度
HIDDEN_SIZE = 256     # RNN隐藏单元数
BATCH_SIZE = 32       # 批量大小
NUM_LAYERS = 1        # RNN层数
EPOCHS = 50           # 训练轮数
LEARNING_RATE = 0.001 # 学习率
GENERATE_LEN = 50     # 续写文本长度
DATA_PATH = './data/timemachine.txt'
DATA_URL = 'https://raw.githubusercontent.com/d2l-ai/d2l-en/master/data/timemachine.txt'


# ==================== 数据加载与预处理 ====================
def download_data():
    """自动下载数据集（如不存在）"""
    if not os.path.exists(DATA_PATH):
        os.makedirs('./data', exist_ok=True)
        print(f"正在下载数据集到 {DATA_PATH}...")
        urllib.request.urlretrieve(DATA_URL, DATA_PATH)
        print("下载完成！")
    else:
        print(f"使用已有数据集: {DATA_PATH}")


def load_text():
    """加载文本并转换为小写"""
    with open(DATA_PATH, 'r', encoding='utf-8') as f:
        text = f.read()
    return text.lower()


def tokenize(text):
    """单词级分词：按非字母字符分割"""
    # 保留单词，去除多余空格和特殊字符
    words = re.findall(r'\b[a-zA-Z]+\b', text)
    return words


# ==================== 词汇表构建 ====================
class Vocabulary:
    """词汇表类：管理单词到索引的映射"""
    def __init__(self):
        self.word2idx = {}
        self.idx2word = {}
        self.unk_token = '<UNK>'
        self.unk_idx = 0
        self.word2idx[self.unk_token] = self.unk_idx
        self.idx2word[self.unk_idx] = self.unk_token
        self.vocab_size = 1
    
    def build_vocab(self, words, min_freq=1):
        """根据词频构建词汇表"""
        word_counts = {}
        for word in words:
            word_counts[word] = word_counts.get(word, 0) + 1
        
        # 按频率排序，过滤低频词
        sorted_words = sorted(
            [(w, c) for w, c in word_counts.items() if c >= min_freq],
            key=lambda x: x[1], reverse=True
        )
        
        for word, _ in sorted_words:
            if word not in self.word2idx:
                self.word2idx[word] = self.vocab_size
                self.idx2word[self.vocab_size] = word
                self.vocab_size += 1
        
        print(f"词汇表大小: {self.vocab_size}")
    
    def encode(self, words):
        """将单词列表转换为索引列表"""
        return [self.word2idx.get(w, self.unk_idx) for w in words]
    
    def decode(self, indices):
        """将索引列表转换为单词列表"""
        return [self.idx2word.get(i, self.unk_token) for i in indices]


# ==================== 滑动窗口数据集 ====================
class TextDataset(Dataset):
    """时序文本数据集：滑动窗口切分"""
    def __init__(self, words, vocab, seq_len):
        self.seq_len = seq_len
        self.vocab = vocab
        
        # 将文本编码为索引
        self.data = vocab.encode(words)
        
        # 滑动窗口切分：总样本数 = 总词数 - seq_len
        self.samples = []
        for i in range(len(self.data) - seq_len):
            x = self.data[i:i + seq_len]
            y = self.data[i + 1:i + seq_len + 1]  # 每个词预测下一个词
            self.samples.append((x, y))
        
        print(f"总样本数: {len(self.samples)}")
    
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        x, y = self.samples[idx]
        return torch.tensor(x, dtype=torch.long), torch.tensor(y, dtype=torch.long)


def collate_fn(batch):
    """自定义batch处理：保持时序顺序"""
    xs = torch.stack([item[0] for item in batch])
    ys = torch.stack([item[1] for item in batch])
    return xs, ys


# ==================== RNN模型定义 ====================
class RNNModel(nn.Module):
    """
    原始RNN文本预测模型
    结构：Embedding → RNN → Linear
    """
    def __init__(self, vocab_size, embedding_dim, hidden_size, num_layers):
        super(RNNModel, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.vocab_size = vocab_size
        
        # 词嵌入层：将词索引映射为稠密向量
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        
        # 原始RNN层：使用nn.RNN，非LSTM/GRU
        self.rnn = nn.RNN(
            input_size=embedding_dim,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            nonlinearity='tanh'  # 原始RNN使用tanh激活
        )
        
        # 全连接输出层：将隐藏状态映射到词汇表空间
        self.fc = nn.Linear(hidden_size, vocab_size)
    
    def forward(self, x, hidden=None):
        """
        前向传播
        x: (batch_size, seq_len) 词索引
        hidden: (num_layers, batch_size, hidden_size) 隐状态
        返回: (batch_size, seq_len, vocab_size) 预测概率分布，以及新隐状态
        """
        batch_size = x.size(0)
        
        # 初始化隐状态（如未提供）
        if hidden is None:
            hidden = self.init_hidden(batch_size, x.device)
        
        # Embedding: (batch, seq_len) -> (batch, seq_len, embedding_dim)
        embedded = self.embedding(x)
        
        # RNN: (batch, seq_len, embedding_dim) -> (batch, seq_len, hidden_size)
        rnn_out, hidden = self.rnn(embedded, hidden)
        
        # 全连接: (batch, seq_len, hidden_size) -> (batch, seq_len, vocab_size)
        output = self.fc(rnn_out)
        
        return output, hidden
    
    def init_hidden(self, batch_size, device):
        """随机初始化隐状态"""
        return torch.zeros(self.num_layers, batch_size, self.hidden_size, device=device)


# ==================== 训练流程 ====================
def train_model(model, dataloader, epochs, device):
    """BPTT训练：沿时间反向传播"""
    model.train()
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
    
    print("\n开始训练...")
    for epoch in range(epochs):
        total_loss = 0
        
        for batch_idx, (x, y) in enumerate(dataloader):
            x, y = x.to(device), y.to(device)
            
            # 每个batch重新初始化hidden，匹配实际batch大小（处理最后一个不完整的batch）
            hidden = model.init_hidden(x.size(0), device)
            
            # 前向传播
            output, hidden = model(x, hidden)
            
            # 分离隐状态，防止反向传播过长（截断BPTT）
            hidden = hidden.detach()
            
            # 计算损失：将输出和标签展平
            # output: (batch, seq_len, vocab_size) -> (batch*seq_len, vocab_size)
            # y: (batch, seq_len) -> (batch*seq_len)
            loss = criterion(output.view(-1, model.vocab_size), y.view(-1))
            
            # 反向传播
            optimizer.zero_grad()
            loss.backward()
            
            # 梯度裁剪，防止梯度爆炸
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5)
            
            optimizer.step()
            
            total_loss += loss.item()
        
        avg_loss = total_loss / len(dataloader)
        if (epoch + 1) % 5 == 0 or epoch == 0:
            print(f"Epoch [{epoch+1}/{epochs}], Loss: {avg_loss:.4f}")
    
    print("训练完成！")


# ==================== 推理与生成 ====================
def generate_text(model, vocab, seed_text, generate_len, device):
    """
    交互式文本生成
    输入种子句子，模型续写指定长度的文本
    """
    model.eval()
    
    # 对输入文本进行编码
    seed_words = tokenize(seed_text.lower())
    if len(seed_words) == 0:
        return "请输入有效的英文句子。"
    
    # 如果种子文本过长，只保留最后seq_len个词
    if len(seed_words) > SEQ_LEN:
        seed_words = seed_words[-SEQ_LEN:]
    
    input_indices = vocab.encode(seed_words)
    input_tensor = torch.tensor([input_indices], dtype=torch.long, device=device)
    
    generated_indices = input_indices.copy()
    
    with torch.no_grad():
        # 初始化隐状态
        hidden = model.init_hidden(1, device)
        
        # 先处理种子文本，建立隐状态
        if len(input_indices) > 0:
            _, hidden = model(input_tensor, hidden)
        
        # 当前输入为最后一个词（或从种子最后一个词开始）
        current_input = torch.tensor([[generated_indices[-1]]], dtype=torch.long, device=device)
        
        # 逐词生成
        for _ in range(generate_len):
            output, hidden = model(current_input, hidden)
            
            # 取最后一个时间步的输出，预测下一个词
            logits = output[0, -1, :]  # (vocab_size,)
            probs = torch.softmax(logits, dim=0)
            
            # 采样生成（使用温度参数调节随机性）
            temperature = 0.8
            probs = probs ** (1 / temperature)
            probs = probs / probs.sum()
            
            # 贪婪解码或采样
            next_idx = torch.multinomial(probs, 1).item()
            # next_idx = torch.argmax(probs).item()  # 贪婪解码
            
            generated_indices.append(next_idx)
            current_input = torch.tensor([[next_idx]], dtype=torch.long, device=device)
    
    # 将索引转换为单词
    generated_words = vocab.decode(generated_indices)
    return ' '.join(generated_words)


# ==================== 主程序 ====================
def main():
    # 设置随机种子，保证可复现
    torch.manual_seed(42)
    np.random.seed(42)
    random.seed(42)
    
    # 设备选择
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"使用设备: {device}")
    
    # 1. 数据加载
    print("\n" + "="*50)
    print("步骤1: 加载数据")
    print("="*50)
    download_data()
    text = load_text()
    words = tokenize(text)
    print(f"文本总词数: {len(words)}")
    print(f"示例前20词: {' '.join(words[:20])}")
    
    # 2. 构建词汇表
    print("\n" + "="*50)
    print("步骤2: 构建词汇表")
    print("="*50)
    vocab = Vocabulary()
    vocab.build_vocab(words, min_freq=2)
    
    # 3. 构造数据集
    print("\n" + "="*50)
    print("步骤3: 构造滑动窗口数据集")
    print("="*50)
    dataset = TextDataset(words, vocab, SEQ_LEN)
    dataloader = DataLoader(
        dataset, 
        batch_size=BATCH_SIZE, 
        shuffle=True,
        collate_fn=collate_fn
    )
    
    # 4. 创建模型
    print("\n" + "="*50)
    print("步骤4: 初始化RNN模型")
    print("="*50)
    model = RNNModel(
        vocab_size=vocab.vocab_size,
        embedding_dim=EMBEDDING_DIM,
        hidden_size=HIDDEN_SIZE,
        num_layers=NUM_LAYERS
    ).to(device)
    print(f"模型结构:\n{model}")
    print(f"总参数量: {sum(p.numel() for p in model.parameters()):,}")
    
    # 5. 训练
    print("\n" + "="*50)
    print("步骤5: 训练模型")
    print("="*50)
    train_model(model, dataloader, EPOCHS, device)
    
    # 6. 交互式预测
    print("\n" + "="*50)
    print("步骤6: 交互式文本生成")
    print("="*50)
    print(f"输入英文句子，模型将续写{GENERATE_LEN}个词")
    print("输入 'quit' 退出程序")
    print("-"*50)
    
    while True:
        user_input = input("\n请输入起始句子: ").strip()
        
        if user_input.lower() == 'quit':
            print("感谢使用，再见！")
            break
        
        if not user_input:
            print("输入不能为空，请重新输入。")
            continue
        
        # 生成续写
        generated = generate_text(model, vocab, user_input, GENERATE_LEN, device)
        print(f"\n生成结果:\n{generated}")


if __name__ == '__main__':
    main()
