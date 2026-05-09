"""
以文本为输入的多分类任务, 实验一下用RNN, LSTM等模型的跑通训练。

模型: Embedding → RNN → 取最后隐藏状态 → Linear → Softmax
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
EMBED_DIM   = 64
HIDDEN_DIM  = 64
LR          = 1e-3
BATCH_SIZE  = 64
EPOCHS      = 20
TRAIN_RATIO = 0.8

random.seed(SEED)
torch.manual_seed(SEED)

# ─── 1. 数据生成 ────────────────────────────────────────────
TEMPLATES = [
    '{}, {}引发关注',
    '最新消息：{}',
    '{}, {}再创纪录',
    '{}, {}成为焦点',
    '今日热点：{}',
    '{}, {}引发热议',
]

CATEGORIES = ['体育', '财经', '娱乐', '科技', '政治']
CATEGORY_TO_ID = {cat: i for i, cat in enumerate(CATEGORIES)}
KEYWORDS_BY_CATEGORY = {
    '体育': ['篮球', '足球', '湖人', 'C罗', '梅西', 'NBA', '世界杯', '进球'],
    '财经': ['股票', '基金', 'A股', '降息', '涨停', '财报', '投资', '黄金'],
    '娱乐': ['电影', '演唱会', '新歌', '综艺', '明星', '票房', '上映', '专辑'],
    '科技': ['AI', '算法', '芯片', '5G', '大模型', '无人驾驶', '苹果', '华为'],
    '政治': ['选举', '议会', '外交', '峰会', '政策', '法案', '会谈', '声明'],
}

def make_sentence():
    ''' 会出错的情况：\n
        '{}, {}, {}引发关注'.format(obj, kw)  # 需要3个, 只给2个 → IndexError \n
        '{}引发关注'.format(obj, kw)          # 需要1个, 给了2个 → 仍然正常(Python 的 str.format() 在参数多于占位符时也不会报错: 只取第一个)
    '''
    CATEGORY = random.choice(CATEGORIES)
    kw   = random.choice(KEYWORDS_BY_CATEGORY[CATEGORY])
    tmpl = random.choice(TEMPLATES)
    obj  = random.choice(KEYWORDS_BY_CATEGORY[CATEGORY])
    # 确保占位符数量匹配
    if tmpl.count('{}') == 1:
        sent = tmpl.format(kw)
    else:
        # 需要两个关键词的情况
        kw2 = random.choice(KEYWORDS_BY_CATEGORY[CATEGORY])
        sent = tmpl.format(kw, kw2)
    
    return (sent, CATEGORY_TO_ID[CATEGORY])

def build_dataset(n=N_SAMPLES):
    data = []
    for _ in range(n // 2):
        data.append(make_sentence())
    random.shuffle(data)

    # print("\n样本示例: ")
    # for sent, cat in data[:5]:
    #     print(f"  [{cat}] {sent}...")

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
    中文关键词分类器（RNN + MaxPooling 版）
    架构: Embedding → RNN → MaxPool → BN → Dropout → Linear → Softmax → (EntropyLoss)
    """
    def __init__(self, vocab_size, embed_dim=EMBED_DIM, hidden_dim=HIDDEN_DIM, dropout=0.3):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=0)
        self.rnn       = nn.RNN(embed_dim, hidden_dim, batch_first=True)
        self.lstm      = nn.LSTM(hidden_dim, hidden_dim, batch_first=True)
        self.bn        = nn.BatchNorm1d(hidden_dim)
        self.dropout   = nn.Dropout(dropout)
        self.fc        = nn.Linear(hidden_dim, 5)  # 5分类

    def forward(self, x):
        # x: (batch, seq_len)
        e, _ = self.rnn(self.embedding(x))  # (B, L, hidden_dim)
        e, _ = self.lstm(e)  # (B, L, hidden_dim)
        pooled = e.max(dim=1)[0]            # (B, hidden_dim)  对序列做 max pooling
        pooled = self.dropout(self.bn(pooled))
        out = self.fc(pooled)  # (B, 5) 交叉熵损失函数内部会自带 softmax，所以这里直接输出 logits 就行了
        return out


# ─── 5. 训练与评估 ──────────────────────────────────────────
def evaluate(model, loader):
    model.eval()
    correct = total = 0
    with torch.no_grad():
        for X, y in loader:
            prob    = model(X)
            pred    = torch.argmax(prob, dim=1)
            # 如果 pred 和 y 一致，就说明预测正确
            correct += (pred == y.long()).sum().item()
            total   += len(y)
    return correct / total


def train():
    print("生成数据集...")
    data  = build_dataset(N_SAMPLES)
    vocab = build_vocab(data)
    print(f"  样本数: {len(data)}, 词表大小: {len(vocab)}")

    split      = int(len(data) * TRAIN_RATIO)
    train_data = data[:split]
    val_data   = data[split:]

    train_loader = DataLoader(TextDataset(train_data, vocab), batch_size=BATCH_SIZE, shuffle=True)
    val_loader   = DataLoader(TextDataset(val_data,   vocab), batch_size=BATCH_SIZE)

    model     = KeywordRNN(vocab_size=len(vocab))
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
    test_sents = [
        # 包含关键词的正样本(有关键词)
        '制裁,调解引发热议',
        '今日热点：脱口秀',
        '田径,游泳再创纪录',
        '楼市,楼市成为焦点',
        'VR,卫星引发关注',

        # 领域外但格式相似(有句子结构)
        '露营装备，市场规模引发热议',
        '最新消息：预制菜迎来新变化',
        '今日热点：人工智能绘画行业动态',
        '冥想APP, 日活跃用户持续走高',
        '最新消息：跨境电商迎来新变化',

        # ❌ 不包含任何 KEYWORDS_BY_CATEGORY 中的关键词
        # ❌ 不匹配 TEMPLATES 的任何格式
        # ✅ 可用于测试分类器对异常输入的鲁棒性
        '最近在学做蛋糕',
        '这家咖啡店环境很好',
        '推荐一本好书给大家',
        '周末去看了一场话剧',
        '猫趴在窗台上晒太阳',
        '今天天气真好',
        '周末计划去公园',
        '刚看了部电影',
        '晚餐吃了火锅',
    ]

    model.eval()
    with torch.no_grad():
        for sent in test_sents:
            ids   = torch.tensor([encode(sent, vocab)], dtype=torch.long)
            prob  = model(ids)
            # label = CATEGORIES[torch.argmax(prob)]
            # print(f"  [{label}({prob})]  {sent}")

            probs = torch.softmax(prob, dim=1)  # 转为概率分布
            label = CATEGORIES[torch.argmax(probs)]
            probs_list = probs.squeeze().tolist()  # 转为 Python list
            print(f"  [{label}]  {sent}")
            print(f"    概率分布: {dict(zip(CATEGORIES, probs_list))}")


if __name__ == '__main__':
    train()
    # build_dataset()
