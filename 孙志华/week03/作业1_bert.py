"""
以文本为输入的多分类任务, 实验一下用RNN, LSTM等模型的跑通训练。
为加强模型的表达能力, 使用预训练的词向量和更深层次的网络结构。

模型: BERT → LSTM → MaxPool → BN → Dropout → Linear
优化: Adam (lr=1e-3)   损失: CrossEntropyLoss

依赖: torch >= 2.0   (pip install torch)
    pip install transformers==5.8.0
    pip install tokenizers==0.22.2

无法访问 Hugging Face 官方服务器时可以使用国内镜像源，以下是设置环境变量的命令：
    需要本地下载 bert-base-chinese 模型并放在当前目录下的 ./bert-base-chinese 文件夹中
    set HF_ENDPOINT=https://hf-mirror.com
    hf download google-bert/bert-base-chinese --local-dir ./bert-base-chinese
"""

import random
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from transformers import BertTokenizer, BertModel

# ─── 超参数 ────────────────────────────────────────────────
SEED        = 42
N_SAMPLES   = 200
MAXLEN      = 32
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
    """生成一条训练数据"""
    category = random.choice(CATEGORIES)
    kw = random.choice(KEYWORDS_BY_CATEGORY[category])
    tmpl = random.choice(TEMPLATES)
    
    if tmpl.count('{}') == 1:
        sent = tmpl.format(kw)
    else:
        kw2 = random.choice(KEYWORDS_BY_CATEGORY[category])
        sent = tmpl.format(kw, kw2)
    
    return (sent, CATEGORY_TO_ID[category])

def build_dataset(n=N_SAMPLES):
    """构建数据集"""
    data = []
    for _ in range(n):
        data.append(make_sentence())
    random.shuffle(data)
    return data

# ─── 2. 使用 BERT Tokenizer 的 Dataset ────────────────────
class TextDataset(Dataset):
    def __init__(self, data, tokenizer, maxlen=MAXLEN):
        self.input_ids = []
        self.attention_masks = []
        self.labels = []
        
        for text, label in data:
            # 使用 BERT tokenizer 进行编码
            encoded = tokenizer(
                text,
                max_length=maxlen,
                padding='max_length',
                truncation=True,
                return_tensors='pt'
            )
            self.input_ids.append(encoded['input_ids'].squeeze(0))
            self.attention_masks.append(encoded['attention_mask'].squeeze(0))
            self.labels.append(torch.tensor(label, dtype=torch.long))
    
    def __len__(self):
        return len(self.labels)
    
    def __getitem__(self, idx):
        return self.input_ids[idx], self.attention_masks[idx], self.labels[idx]

# ─── 3. 模型定义 ────────────────────────────────────────────
class KeywordClassifier(nn.Module):
    """
    中文关键词分类器
    架构: BERT → LSTM → MaxPool → BN → Dropout → Linear
    """
    def __init__(self, hidden_dim=HIDDEN_DIM, dropout=0.3):
        super().__init__()
        # 加载 BERT 模型
        self.bert = BertModel.from_pretrained('./bert-base-chinese')
        bert_dim = self.bert.config.hidden_size  # 768
        
        # LSTM 层
        self.lstm = nn.LSTM(
            bert_dim, 
            hidden_dim, 
            batch_first=True,
            bidirectional=False
        )
        
        # 全连接层
        self.bn = nn.BatchNorm1d(hidden_dim)
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(hidden_dim, 5)
        
        # 冻结 BERT（可选，加快训练）
        for param in self.bert.parameters():
            param.requires_grad = False
    
    def forward(self, input_ids, attention_mask):
        # BERT 编码
        with torch.no_grad():
            bert_outputs = self.bert(
                input_ids=input_ids, 
                attention_mask=attention_mask
            )
            bert_out = bert_outputs.last_hidden_state  # (batch, seq_len, 768)
        
        # LSTM
        lstm_out, _ = self.lstm(bert_out)  # (batch, seq_len, hidden_dim)
        
        # Max Pooling
        pooled = lstm_out.max(dim=1)[0]  # (batch, hidden_dim)
        
        # 分类头
        pooled = self.bn(pooled)
        pooled = self.dropout(pooled)
        logits = self.fc(pooled)
        
        return logits

# ─── 4. 训练与评估 ──────────────────────────────────────────
def evaluate(model, loader):
    """评估函数"""
    model.eval()
    correct = total = 0
    with torch.no_grad():
        for input_ids, attention_mask, labels in loader:
            logits = model(input_ids, attention_mask)
            preds = torch.argmax(logits, dim=1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)
    return correct / total

def train():
    print("=" * 50)
    print("开始训练...")
    print("=" * 50)
    
    # 1. 生成数据
    print("\n1. 生成数据集...")
    data = build_dataset(N_SAMPLES)
    print(f"   样本数: {len(data)}")
    
    # 显示几个样本
    print("\n   样本示例:")
    for i in range(3):
        text, label = data[i]
        print(f"     [{CATEGORIES[label]}] {text}")
    
    # 2. 加载 BERT Tokenizer
    print("\n2. 加载 BERT Tokenizer...")
    tokenizer = BertTokenizer.from_pretrained('./bert-base-chinese')
    print("   Tokenizer 加载完成")
    
    # 3. 划分数据集
    split = int(len(data) * TRAIN_RATIO)
    train_data = data[:split]
    val_data = data[split:]
    print(f"\n3. 数据集划分: 训练集={len(train_data)}, 验证集={len(val_data)}")
    
    # 4. 创建 DataLoader
    train_dataset = TextDataset(train_data, tokenizer)
    val_dataset = TextDataset(val_data, tokenizer)
    
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE)
    
    # 5. 创建模型
    print("\n4. 创建模型...")
    model = KeywordClassifier()
    
    # 计算参数量
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"   总参数量: {total_params:,}")
    print(f"   可训练参数量: {trainable_params:,}")
    print(f"   BERT 冻结: {trainable_params < total_params}")
    
    # 6. 训练配置
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=LR)
    
    # 7. 训练循环
    print("\n5. 开始训练...")
    print("-" * 50)
    
    best_acc = 0.0
    for epoch in range(1, EPOCHS + 1):
        model.train()
        total_loss = 0.0
        
        for input_ids, attention_mask, labels in train_loader:
            # 前向传播
            logits = model(input_ids, attention_mask)
            loss = criterion(logits, labels)
            
            # 反向传播
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
        
        # 验证
        avg_loss = total_loss / len(train_loader)
        val_acc = evaluate(model, val_loader)
        
        # 保存最佳模型
        if val_acc > best_acc:
            best_acc = val_acc
        
        print(f"Epoch {epoch:2d}/{EPOCHS} | Loss: {avg_loss:.4f} | Val Acc: {val_acc:.4f} | Best: {best_acc:.4f}")
    
    print("-" * 50)
    print(f"\n✅ 训练完成！最佳验证准确率: {best_acc:.4f}")
    
    # 8. 推理测试
    print("\n" + "=" * 50)
    print("推理测试")
    print("=" * 50)
    
    test_sents = [
        # 包含关键词的正样本
        ('制裁,调解引发热议', '预期: 政治'),
        ('今日热点：脱口秀', '预期: 娱乐'),
        ('田径,游泳再创纪录', '预期: 体育'),
        ('楼市,楼市成为焦点', '预期: 财经'),
        ('VR,卫星引发关注', '预期: 科技'),
        
        # 边界样本
        ('最新消息：预制菜迎来新变化', '预期: 其他'),
        ('今日热点：人工智能绘画行业动态', '预期: 科技'),
        ('冥想APP, 日活跃用户持续走高', '预期: 科技'),
        
        # 无关键词样本
        ('最近在学做蛋糕', '预期: 其他'),
        ('今天天气真好', '预期: 其他'),
        ('周末计划去公园', '预期: 其他'),
    ]
    
    model.eval()
    with torch.no_grad():
        for sent, expected in test_sents:
            # 编码
            encoded = tokenizer(
                sent,
                max_length=MAXLEN,
                padding='max_length',
                truncation=True,
                return_tensors='pt'
            )
            
            # 预测
            logits = model(encoded['input_ids'], encoded['attention_mask'])
            probs = torch.softmax(logits, dim=1)
            pred_label = CATEGORIES[torch.argmax(probs).item()]
            
            # 显示结果
            probs_dict = {cat: f"{p:.3f}" for cat, p in zip(CATEGORIES, probs.squeeze().tolist())}
            print(f"\n文本: {sent}")
            print(f"预测: {pred_label}")
            print(f"概率: {probs_dict}")

if __name__ == '__main__':
    train()