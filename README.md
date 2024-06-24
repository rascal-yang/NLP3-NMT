# Neural Machine Translation, NMT

## 项目名称
文言文翻译与判别系统

## 简介
本项目旨在实现文言文与现代汉语之间的自动翻译，并能够对文本进行判别，确定其为文言文或现代汉语。项目包含三个Python脚本：`Modern2Ancient.py`、`Ancient2Modern.py` 和 `Ancient_judge.py`，分别对应现代文到文言文的翻译、文言文到现代文的翻译以及文本类型的判别。

## 模块介绍

### 1. 现代文到文言文翻译模块 (`Modern2Ancient.py`)
- **模型**: 使用基于BERT的Transformer翻译模型。
- **功能**: 接受现代汉语输入，输出对应的文言文表达。

### 2. 文言文到现代文翻译模块 (`Ancient2Modern.py`)
- **模型**: 同样采用基于BERT的Transformer翻译模型。
- **功能**: 将文言文文本转换为现代汉语，增加理解性。

### 3. 文本类型判别模块 (`Ancient_judge.py`)
- **模型**: 结合了BERT嵌入和双向LSTM的注意力机制分类器。
- **功能**: 判断给定文本是文言文还是现代汉语，并输出判别结果。

## 环境要求
- Python 3.9
- PyTorch cuda
- Transformers 2.11+
- NLTK (仅 `Ancient2Modern.py` 中用于计算BLEU分数)
- tqdm (用于命令行进度条显示)

## 安装指南
1. 克隆项目到本地机器
   ```bash
   git clone https://github.com/rascal-yang/NLP3-NMT.git
   ```
2. 创建并激活虚拟环境（推荐）
   ```
   python -m venv venv
   source venv/bin/activate  # 对于Windows使用 `venv\Scripts\activate`
   ```
3. 安装依赖
   ```
   pip install torch transformers nltk tqdm
   ```

## 使用方法
### 现代文到文言文
- 运行 `Modern2Ancient.py` 脚本：
  ```bash
  python Modern2Ancient.py
  ```
- 使用 `predictM2A` 函数进行翻译：
  ```python
  from Modern2Ancient import predictM2A
  sentence = "自从文字被发明以来，能够了解的上古人物，就是经传中所提到的。"
  print(predictM2A(sentence))
  ```

### 文言文到现代文
- 运行 `Ancient2Modern.py` 脚本：
  ```bash
  python Ancient2Modern.py
  ```
- 使用 `predictA2M` 函数进行翻译：
  ```python
  from Ancient2Modern import predictA2M
  sentence = "假之以便，唆之使前，断其援应，陷之死地。"
  print(predictA2M(sentence))
  ```

### 文本类型判别
- 运行 `Ancient_judge.py` 脚本：
  ```bash
  python Ancient_judge.py
  ```
- 使用 `predict_sentence` 函数进行判别：
  ```python
  from Ancient_judge import an_judge
  sentence = "我爱中国"
  print(an_judge.predict_sentence(sentence))
  ```

## 数据集
项目数据集源自于 https://github.com/NiuTrans/Classical-Modern ，其提供了
一套极为丰富的文言文与现代文对照语料库。

## 模型介绍

### 1. 现代文到文言文翻译模块 (`Modern2Ancient.py`)
- **模型架构**: Transformer-based Sequence-to-Sequence (Seq2Seq) model with BERT as the encoder.
- **关键技术**: Attention mechanism, Transformer Decoder.
- **特点**: 利用BERT的深层语义理解能力，结合Transformer的长距离依赖捕捉能力，实现高质量的翻译输出。

### 2. 文言文到现代文翻译模块 (`Ancient2Modern.py`)
- **模型架构**: 与`Modern2Ancient.py`相同，也是基于BERT的Transformer Seq2Seq模型。
- **关键技术**: 同上，包括注意力机制和Transformer解码器技术。


### 3. 文本类型判别模块 (`Ancient_judge.py`)
- **模型架构**: BERT Embeddings + Bidirectional LSTM + Attention Mechanism + Fully Connected Layer.
- **关键技术**: BERT用于文本嵌入，双向LSTM捕捉前后文信息，注意力机制聚焦文本关键部分，全连接层用于分类。
- **特点**: 结合了深度学习中的多种技术，通过注意力机制强化模型对文本中关键信息的识别，提高判别准确性。

## 模型训练
- 每个脚本均包含训练函数，可以对模型进行训练和优化。

## 注意事项
- 确保输入文本长度不超过模型的最大序列长度限制。
- 模型可能需要根据特定数据集进行微调以获得最佳效果。
- 本项目使用了BERT-Ancient-Chinese预训练模型，可以使用代码从hugging face上下载
