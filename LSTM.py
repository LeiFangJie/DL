"""
LSTM文本生成模型 - 基于Time Machine数据集
单词级文本预测，使用LSTM实现
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
import matplotlib.pyplot as plt
from collections import deque


# ==================== 超参数配置 ====================
SEQ_LEN = 64          # 序列长度
EMBEDDING_DIM = 128   # 词向量维度
HIDDEN_SIZE = 512     # LSTM隐藏单元数
BATCH_SIZE = 64       # 批量大小
NUM_LAYERS = 3        # LSTM层数
EPOCHS = 50           # 训练轮数
LEARNING_RATE = 0.001 # 学习率
GENERATE_LEN = 100    # 续写文本长度
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
        self.data = vocab.encode(words)
        
        self.samples = []
        for i in range(len(self.data) - seq_len):
            x = self.data[i:i + seq_len]
            y = self.data[i + 1:i + seq_len + 1]
            self.samples.append((x, y))
        
        print(f"总样本数: {len(self.samples)}")
    
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        x, y = self.samples[idx]
        return torch.tensor(x, dtype=torch.long), torch.tensor(y, dtype=torch.long)


def collate_fn(batch):
    """自定义batch处理"""
    xs = torch.stack([item[0] for item in batch])
    ys = torch.stack([item[1] for item in batch])
    return xs, ys


# ==================== LSTM模型定义 ====================
class LSTMModel(nn.Module):
    """
    LSTM文本预测模型
    结构：Embedding → LSTM → Linear
    LSTM包含三重门控：遗忘门、输入门、输出门
    """
    def __init__(self, vocab_size, embedding_dim, hidden_size, num_layers):
        super(LSTMModel, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.vocab_size = vocab_size
        
        # 词嵌入层
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        
        # LSTM层：包含遗忘门、输入门、输出门三重门控
        self.lstm = nn.LSTM(
            input_size=embedding_dim,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=0.3 if num_layers > 1 else 0
        )
        
        # 全连接输出层
        self.fc = nn.Linear(hidden_size, vocab_size)
    
    def forward(self, x, hidden=None):
        """
        前向传播
        x: (batch_size, seq_len) 词索引
        hidden: (h, c) 元组，各为 (num_layers, batch_size, hidden_size)
        返回: 输出概率分布，以及新的(h, c)元组
        """
        batch_size = x.size(0)
        
        # 初始化隐状态（如未提供）
        if hidden is None:
            hidden = self.init_hidden(batch_size, x.device)
        
        # Embedding
        embedded = self.embedding(x)
        
        # LSTM: 返回输出和(h_n, c_n)元组
        lstm_out, (h_n, c_n) = self.lstm(embedded, hidden)
        
        # 全连接
        output = self.fc(lstm_out)
        
        return output, (h_n, c_n)
    
    def init_hidden(self, batch_size, device):
        """
        初始化LSTM的双重状态
        h: 隐藏状态 (short-term memory)
        c: 细胞状态 (long-term memory)
        """
        h = torch.zeros(self.num_layers, batch_size, self.hidden_size, device=device)
        c = torch.zeros(self.num_layers, batch_size, self.hidden_size, device=device)
        return (h, c)


# ==================== 训练流程 ====================
def train_model(model, dataloader, epochs, device):
    """BPTT训练：沿时间反向传播"""
    model.train()
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
    
    plt.ion()
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Loss')
    ax.set_title('Training Loss (Real-time)')
    line, = ax.plot([], [], 'b-', linewidth=2)
    loss_history = deque(maxlen=epochs)
    epoch_list = deque(maxlen=epochs)
    
    print("\n开始训练...")
    print("训练过程中会显示实时loss曲线窗口...")
    
    for epoch in range(epochs):
        total_loss = 0
        
        for batch_idx, (x, y) in enumerate(dataloader):
            x, y = x.to(device), y.to(device)
            
            # 初始化LSTM双重状态
            hidden = model.init_hidden(x.size(0), device)
            
            # 前向传播
            output, (h_n, c_n) = model(x, hidden)
            
            # 分离双重状态（截断BPTT）
            h_n = h_n.detach()
            c_n = c_n.detach()
            
            # 计算损失
            loss = criterion(output.view(-1, model.vocab_size), y.view(-1))
            
            # 反向传播
            optimizer.zero_grad()
            loss.backward()
            
            # 梯度裁剪
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5)
            
            optimizer.step()
            
            total_loss += loss.item()
        
        avg_loss = total_loss / len(dataloader)
        
        loss_history.append(avg_loss)
        epoch_list.append(epoch + 1)
        
        line.set_xdata(list(epoch_list))
        line.set_ydata(list(loss_history))
        ax.relim()
        ax.autoscale_view()
        ax.set_title(f'Training Loss (Real-time) - Epoch {epoch+1}/{epochs}')
        plt.draw()
        plt.pause(0.001)
        
        if (epoch + 1) % 5 == 0 or epoch == 0:
            print(f"Epoch [{epoch+1}/{epochs}], Loss: {avg_loss:.4f}")
    
    plt.ioff()
    print("训练完成！关闭loss曲线窗口后继续...")
    plt.show()


# ==================== 原文比对功能 ====================
ORIGINAL_TEXT = ""

def check_originality(generated_words, min_match_len=5):
    """检测生成文本与原文的重复情况"""
    if not ORIGINAL_TEXT or len(generated_words) < min_match_len:
        return 100.0, []
    
    original_words = ORIGINAL_TEXT.lower().split()
    gen_lower = [w.lower() for w in generated_words]
    
    matched_positions = set()
    repeated_phrases = []
    
    for n in range(min(min_match_len + 10, len(gen_lower)), min_match_len - 1, -1):
        for i in range(len(gen_lower) - n + 1):
            if i in matched_positions:
                continue
            
            ngram = gen_lower[i:i+n]
            
            for j in range(len(original_words) - n + 1):
                if original_words[j:j+n] == ngram:
                    for k in range(i, i+n):
                        matched_positions.add(k)
                    phrase = ' '.join(generated_words[i:i+n])
                    repeated_phrases.append((i, n, phrase))
                    break
    
    originality = (1 - len(matched_positions) / len(gen_lower)) * 100 if gen_lower else 100
    
    seen = set()
    unique_phrases = []
    for pos, length, phrase in sorted(repeated_phrases, key=lambda x: (x[0], -x[1])):
        if phrase.lower() not in seen:
            seen.add(phrase.lower())
            unique_phrases.append((pos, length, phrase))
    
    return originality, unique_phrases


def highlight_repeats(generated_words, repeated_phrases):
    """高亮显示重复片段 [重复内容]"""
    if not repeated_phrases:
        return ' '.join(generated_words)
    
    repeat_positions = set()
    for pos, length, _ in repeated_phrases:
        for i in range(pos, pos + length):
            repeat_positions.add(i)
    
    result = []
    in_repeat = False
    
    for i, word in enumerate(generated_words):
        if i in repeat_positions:
            if not in_repeat:
                result.append('[')
                in_repeat = True
        else:
            if in_repeat:
                result.append(']')
                in_repeat = False
        result.append(word)
    
    if in_repeat:
        result.append(']')
    
    text = ' '.join(result)
    text = text.replace(' [ ', ' [').replace(' ] ', '] ')
    return text


# ==================== 推理与生成 ====================
def generate_text(model, vocab, seed_text, generate_len, device):
    """交互式文本生成"""
    model.eval()
    
    seed_words = tokenize(seed_text.lower())
    if len(seed_words) == 0:
        return "请输入有效的英文句子。"
    
    if len(seed_words) > SEQ_LEN:
        seed_words = seed_words[-SEQ_LEN:]
    
    input_indices = vocab.encode(seed_words)
    input_tensor = torch.tensor([input_indices], dtype=torch.long, device=device)
    
    generated_indices = input_indices.copy()
    
    with torch.no_grad():
        # 初始化LSTM双重状态
        hidden = model.init_hidden(1, device)
        
        # 处理种子文本
        if len(input_indices) > 0:
            _, hidden = model(input_tensor, hidden)
        
        current_input = torch.tensor([[generated_indices[-1]]], dtype=torch.long, device=device)
        
        # 逐词生成
        for _ in range(generate_len):
            output, hidden = model(current_input, hidden)
            
            logits = output[0, -1, :]
            probs = torch.softmax(logits, dim=0)
            
            temperature = 0.8
            probs = probs ** (1 / temperature)
            probs = probs / probs.sum()
            
            next_idx = torch.multinomial(probs, 1).item()
            
            generated_indices.append(next_idx)
            current_input = torch.tensor([[next_idx]], dtype=torch.long, device=device)
    
    generated_words = vocab.decode(generated_indices)
    
    originality, repeated_phrases = check_originality(generated_words, min_match_len=4)
    
    highlighted = highlight_repeats(generated_words, repeated_phrases)
    
    result_lines = [highlighted]
    result_lines.append(f"\n[原创度分析] 原创度: {originality:.1f}%")
    
    if repeated_phrases:
        result_lines.append(f"发现 {len(repeated_phrases)} 处与原文重复:")
        for i, (_, _, phrase) in enumerate(repeated_phrases[:5], 1):
            result_lines.append(f"  {i}. \"{phrase}\"")
        if len(repeated_phrases) > 5:
            result_lines.append(f"  ... 还有 {len(repeated_phrases)-5} 处重复")
    else:
        result_lines.append("未发现与原文的连续重复片段")
    
    return '\n'.join(result_lines)


# ==================== 主程序 ====================
def main():
    torch.manual_seed(42)
    np.random.seed(42)
    random.seed(42)
    
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
    
    global ORIGINAL_TEXT
    ORIGINAL_TEXT = ' '.join(words)
    
    # 2. 构建词汇表
    print("\n" + "="*50)
    print("步骤2: 构建词汇表")
    print("="*50)
    vocab = Vocabulary()
    vocab.build_vocab(words, min_freq=1)
    
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
    print("步骤4: 初始化LSTM模型")
    print("="*50)
    model = LSTMModel(
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
        
        generated = generate_text(model, vocab, user_input, GENERATE_LEN, device)
        print(f"\n生成结果:\n{generated}")


if __name__ == '__main__':
    main()
