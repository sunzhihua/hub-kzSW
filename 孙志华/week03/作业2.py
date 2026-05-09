"""
对一个任意包含“你”字的五个字的文本，“你”在第几位，就属于第几类。

模型: Embedding → RNN → LSTM → 取最后隐藏状态 → Linear → Sigmoid
优化: Adam (lr=1e-3)   损失: CrossEntropyLoss   无需 GPU, CPU 即可运行

依赖: torch >= 2.0   (pip install torch)
"""

import random
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

# ─── 超参数 ────────────────────────────────────────────────
SEED        = 42
N_SAMPLES   = 4000
MAXLEN      = 32
EMBED_DIM   = 32
HIDDEN_DIM  = 32
LR          = 1e-3
BATCH_SIZE  = 64
EPOCHS      = 15
TRAIN_RATIO = 0.8

random.seed(SEED)
torch.manual_seed(SEED)

# ─── 1. 数据生成 ────────────────────────────────────────────
TEMPLATES = ['确实问赞的',
            '真聪明去想',
            '太好什了啊',
            '去很说我呀',
            '就喜欢是仨']

other_chars = ['好', '很', '真', '的', '了', '吗', '吧', '啊', '哦', '嗯',
            '我', '他', '她', '它', '这', '那', '有', '在', '是', '不']

def make_sentence(): 
    # 随机选择'你'的位置（1-5，转换为0索引）
    length = 5
    pos = random.randint(0, length - 1)
    
    chars = []
    for i in range(length):
        if i == pos:
            chars.append('你')
        else:
            chars.append(random.choice(other_chars))
    
    sent = ''.join(chars)
    return (sent, pos)  # pos: 0-4

def build_dataset(n=N_SAMPLES):
    data = []
    for _ in range(n):
        data.append(make_sentence())
    random.shuffle(data)
    print(f"样本示例: {data[:20]}")
    return data


# ─── 2. 词表构建与编码 ──────────────────────────────────────
def build_vocab(data):
    vocab = {'<PAD>': 0, '<UNK>': 1}
    for sent, _ in data:
        for ch in sent:
            if ch not in vocab:
                vocab[ch] = len(vocab)
    return vocab


def encode(sent, vocab, maxlen=MAXLEN):
    ids  = [vocab.get(ch, 1) for ch in sent]
    ids  = ids[:maxlen]
    ids += [0] * (maxlen - len(ids))
    return ids


# ─── 3. Dataset / DataLoader ────────────────────────────────
class TextDataset(Dataset):
    def __init__(self, data, vocab):
        self.X = [encode(s, vocab) for s, _ in data]
        self.y = [lb for _, lb in data]

    def __len__(self):
        return len(self.y)

    def __getitem__(self, i):
        return (
            torch.tensor(self.X[i], dtype=torch.long),
            torch.tensor(self.y[i], dtype=torch.long),
        )


# ─── 4. 模型定义 ────────────────────────────────────────────
class KeywordRNN(nn.Module):
    """
    中文关键词分类器 (Embedding + RNN) 
    架构: Embedding → RNN → BN → Dropout → Linear → (CrossEntropyLoss)
    """
    def __init__(self, vocab_size, embed_dim=EMBED_DIM, hidden_dim=HIDDEN_DIM, dropout=0.3):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=0)
        self.rnn       = nn.RNN(embed_dim, hidden_dim, batch_first=True)
        self.bn        = nn.BatchNorm1d(hidden_dim)
        self.dropout   = nn.Dropout(dropout)
        self.fc        = nn.Linear(hidden_dim, 5)

    def forward(self, x):
        # x: (batch, seq_len)
        e, _ = self.rnn(self.embedding(x))  # (B, L, hidden_dim)
        pooled = self.dropout(e[:, -1, :])
        out = self.fc(pooled)               # (B, 5)
        return out

class KeywordRNN_v2(nn.Module):
    """
    中文关键词分类器 (Embedding + RNN + LSTM)
    架构: Embedding → RNN → LSTM → BN → Dropout → Linear → (CrossEntropyLoss)

    为什么 RNN+LSTM 更差？
        过度复杂：模型容量远超任务需求（小任务不需要大模型）
        过拟合：4000个样本不足以训练35k参数
        梯度问题：更深的网络梯度传播更困难
        信息稀释：每层都会损失一些信息
        训练困难：损失平面更崎岖，难以找到最优解
    黄金法则：
        从最简单的模型开始，只有在必要时才增加复杂度
        对于你的任务：单层双向LSTM（或RNN）就完全足够了，再加层只会让效果变差！
    """
    def __init__(self, vocab_size, embed_dim=EMBED_DIM, hidden_dim=HIDDEN_DIM, dropout=0.3):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=0)
        self.rnn       = nn.RNN(embed_dim, hidden_dim, batch_first=True)
        self.lstm      = nn.LSTM(hidden_dim, hidden_dim, batch_first=True)
        self.bn        = nn.BatchNorm1d(hidden_dim)
        self.dropout   = nn.Dropout(dropout)
        self.fc        = nn.Linear(hidden_dim, 5)

    def forward(self, x):
        # x: (batch, seq_len)
        e, _ = self.rnn(self.embedding(x))  # (B, L, hidden_dim)
        e, _ = self.lstm(e)  # (B, L, hidden_dim)

        # pooled = e.max(dim=1)[0]            # (B, hidden_dim)  对序列做 max pooling
        # pooled = self.bn(e[:, -1, :])  # 取最后一个时间步的隐藏状态做分类
        
        pooled = self.dropout(e[:, -1, :])
        out = self.fc(pooled)               # (B, 5)
        return out

class KeywordRNN_v3(nn.Module):  
    """中文关键词分类器 (Embedding + LSTM)    x 
    架构: Embedding → LSTM → Dropout → Linear → (CrossEntropyLoss)"""
    def __init__(self, vocab_size, embed_dim=EMBED_DIM, hidden_dim=HIDDEN_DIM, dropout=0.3):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=0)
        self.lstm = nn.LSTM(embed_dim, hidden_dim, batch_first=True, bidirectional=True)
        self.bn = nn.BatchNorm1d(hidden_dim * 2)
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(hidden_dim * 2, 5)  # * 2 for bidirectional
    
    def forward(self, x):
        emb = self.embedding(x)                    # (B, L, embed_dim)
        lstm_out, (hidden, cell) = self.lstm(emb)  # hidden: (1, B, hidden_dim)
        # pooled = lstm_out.max(dim=1)[0]            # (B, hidden_dim)  对序列做 max pooling
        pooled = lstm_out[:, -1, :]  # 取最后一个时间步的隐藏状态做分类
        pooled = self.dropout(self.bn(pooled))
        out = self.fc(pooled)                     # (B, 5)
        return out

class KeywordRNN_v4(nn.Module):  
    """中文关键词分类器 (Embedding + 双向LSTM) 
    架构: Embedding → LSTM → BN → Dropout → Linear → (CrossEntropyLoss)"""
    """直接识别'你'位置的模型"""
    def __init__(self, vocab_size, embed_dim=32, hidden_dim=64):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=0)
        self.lstm = nn.LSTM(embed_dim, hidden_dim, batch_first=True, bidirectional=True)
        self.fc = nn.Linear(hidden_dim * 2, 5)
        self.dropout = nn.Dropout(0.3)
        
    def forward(self, x):
        emb = self.embedding(x)                    # (B, L, embed_dim)
        lstm_out, (hidden, cell) = self.lstm(emb)  # hidden: (2, B, hidden_dim)
        
        # 拼接双向的最后一个隐藏状态
        hidden_forward = hidden[-2, :, :]  # 正向最后时刻
        hidden_backward = hidden[-1, :, :]  # 反向最后时刻
        hidden_concat = torch.cat([hidden_forward, hidden_backward], dim=1)
        
        out = self.dropout(hidden_concat)
        out = self.fc(out)
        return out

# ─── 5. 训练与评估 ──────────────────────────────────────────
def evaluate(model, loader):
    model.eval()
    correct = total = 0
    with torch.no_grad():
        for X, y in loader:
            prob    = model(X)
            pred    = prob.argmax(dim=1)
            correct += (pred == y.long()).sum().item()
            total   += len(y)
    return correct / total


def train():
    print("生成数据集...")
    data  = build_dataset(N_SAMPLES)
    vocab = build_vocab(data)
    print(f"  样本数: {len(data)}，词表大小: {len(vocab)}")

    split      = int(len(data) * TRAIN_RATIO)
    train_data = data[:split]
    val_data   = data[split:]

    train_loader = DataLoader(TextDataset(train_data, vocab), batch_size=BATCH_SIZE, shuffle=True)
    val_loader   = DataLoader(TextDataset(val_data,   vocab), batch_size=BATCH_SIZE)

    model     = KeywordRNN_v4(vocab_size=len(vocab))
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=LR)

    total_params = sum(p.numel() for p in model.parameters())
    print(f"  模型参数量: {total_params:,}\n")

    for epoch in range(1, EPOCHS + 1):
        model.train()
        total_loss = 0.0
        for X, y in train_loader:
            pred = model(X)
            loss = criterion(pred, y)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        avg_loss = total_loss / len(train_loader)
        val_acc  = evaluate(model, val_loader)
        print(f"Epoch {epoch:2d}/{EPOCHS}  loss={avg_loss:.4f}  val_acc={val_acc:.4f}")

    print(f"\n最终验证准确率: {evaluate(model, val_loader):.4f}")

    print("\n--- 推理示例 ---")
    model.eval()
    test_sents = [
        '你这款产品',
        '今天你出门',
        '下次你还来',
        '等了,你很',
        '1你2水电安装',
        '你ssss',
        '你啊123',
        '啊是你12',
        '真去啊你1',
        '早上好啊你',

        'saqw你',
        '没有关键词',
    ]
    with torch.no_grad():
        for sent in test_sents:
            ids   = torch.tensor([encode(sent, vocab)], dtype=torch.long)
            prob  = model(ids)
            label = prob.argmax().item() + 1
            print(f"  human[{label}({prob})]  {sent}")


if __name__ == '__main__':
    train()


'''
四个模型架构对比表
┌─────────────┬────────────────────────┬──────────────────────────┬──────────────────────────┬──────────────────────────┐
│ 特性        │ KeywordRNN (v1)        │ KeywordRNN_v2            │ KeywordRNN_v3            │ KeywordRNN_v4            │
├─────────────┼────────────────────────┼──────────────────────────┼──────────────────────────┼──────────────────────────┤
│ 架构        │Embedding → RNN → Linear│ Embedding → RNN → LSTM → │ Embedding → 双向LSTM →   │ Embedding → 双向LSTM →   │
│             │                        │ Linear                   │ Linear                   │ Linear                   │
├─────────────┼────────────────────────┼──────────────────────────┼──────────────────────────┼──────────────────────────┤
│ 循环层      │ 单层RNN                 │ RNN + LSTM堆叠            │ 单层双向LSTM             │ 单层双向LSTM             │
├─────────────┼────────────────────────┼──────────────────────────┼──────────────────────────┼──────────────────────────┤
│ 双向性      │ ❌ 单向                 │ ❌ 单向                 │ ✅ 双向                  │ ✅ 双向                  │
├─────────────┼────────────────────────┼──────────────────────────┼──────────────────────────┼──────────────────────────┤
│ 特征聚合    │ 最后时间步               │ 最后时间步                │ 最后时间步                │ 拼接双向隐藏状态         │
├─────────────┼────────────────────────┼──────────────────────────┼──────────────────────────┼──────────────────────────┤
│ 归一化      │ BatchNorm1d             │ BatchNorm1d              │ BatchNorm1d              │ ❌ 无BN                  │
├─────────────┼────────────────────────┼──────────────────────────┼──────────────────────────┼──────────────────────────┤
│ Dropout     │ ✅ 0.3                 │ ✅ 0.3                  │ ✅ 0.3                   │ ✅ 0.3                   │
├─────────────┼─────────────────────────┼─────────────────────────┼──────────────────────────┼──────────────────────────┤
│全连接输入维度│ hidden_dim (32)          │ hidden_dim (32)         │ hidden_dim × 2 (64)      │ hidden_dim × 2 (64)      │
├─────────────┼─────────────────────────┼─────────────────────────┼──────────────────────────┼──────────────────────────┤
│ 参数量      │ ~9,000                  │ ~13,000                  │ ~12,000                  │ ~12,000                  │
├─────────────┼────────────────────────┼──────────────────────────┼──────────────────────────┼──────────────────────────┤
│ 复杂度      │ 低                      │ 高                       │ 中                       │ 中                       │
├─────────────┼────────────────────────┼──────────────────────────┼──────────────────────────┼──────────────────────────┤
│ 预期准确率   │ 75-85%                 │ 20-30%                   │ 60-75%                   │ 90-95%+                  │
├─────────────┼────────────────────────┼──────────────────────────┼──────────────────────────┼──────────────────────────┤
│ 推荐度      │ ⭐⭐⭐                │ ⭐                      │ ⭐⭐                     │ ⭐⭐⭐⭐⭐            │
└─────────────┴────────────────────────┴──────────────────────────┴──────────────────────────┴──────────────────────────┘

    总结：
    - v1: 最简RNN, 训练快, 效果尚可
    - v2: 过度复杂，信息稀释严重，效果最差
    - v3: 双向LSTM但特征聚合方式不佳
    - v4: 最佳设计，充分利用双向信息，推荐使用
'''