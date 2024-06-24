import torch
from torch import nn
from torch.nn import TransformerDecoder, TransformerDecoderLayer
from transformers import AutoTokenizer, AutoModel
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
from torch.optim import SGD
# Check for CUDA availability
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
checkpointpath = "../checkpoint/m2a_checkpoint11.pth"

# 数据集类
class TranslationDataset(Dataset):
    def __init__(self, source_texts, target_texts, tokenizer, max_length=128):
        self.source_texts = source_texts
        self.target_texts = target_texts
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.target_texts)

    def __getitem__(self, idx):
        # 获取索引 idx 对应的源文本和目标文本
        source_text = self.source_texts[idx]
        target_text = self.target_texts[idx]
        
        # 对源文本进行编码
        source_encoded = self.tokenizer(source_text, return_tensors='pt', padding='max_length',
                                        truncation=True, max_length=self.max_length)
        # 对目标文本进行编码
        target_encoded = self.tokenizer(target_text, return_tensors='pt', padding='max_length',
                                        truncation=True, max_length=self.max_length)
        
        # 获取编码后的 input_ids 和 attention mask，并去掉多余的维度
        # input_ids对应中文的编码，attention mask对应中文的注意力掩码 一般为1*len
        source_ids = source_encoded['input_ids'][0]
        target_ids = target_encoded['input_ids'][0]
        source_mask = source_encoded['attention_mask'][0]
        target_mask = target_encoded['attention_mask'][0]
        
        return source_ids, source_mask, target_ids, target_mask
# 机器翻译模型
class TransformerTranslator(nn.Module):
    def __init__(self, bert_model, vocab_size, d_model=768, nhead=8, num_decoder_layers=6):
        super(TransformerTranslator, self).__init__()
        self.bert_model = bert_model
        decoder_layer = TransformerDecoderLayer(d_model, nhead)
        self.transformer_decoder = TransformerDecoder(decoder_layer, num_decoder_layers)
        # 初始化线性层
        self.linear = nn.Linear(d_model, vocab_size)
        
        # 设置权重为 BERT 嵌入层的转置
        self.linear.weight = nn.Parameter(self.bert_model.embeddings.word_embeddings.weight)
        
        # 冻结线性层的参数
        for param in self.linear.parameters():
            param.requires_grad = False

    def generate_square_subsequent_mask(self, sz):
        mask = torch.triu(torch.ones(sz, sz), diagonal=1).bool()
        return mask
    
    def forward(self, source_ids, source_mask, target_ids, target_mask):
        with torch.no_grad():  # 冻结编码器部分
            encoder_outputs = self.bert_model(input_ids=source_ids, attention_mask=source_mask).last_hidden_state
        tgt_embedding = self.bert_model.embeddings(target_ids)
        
        # 转置操作是为了符合 Transformer 解码器的输入要求
        # 转换维度为 [seq_len, batch_size, hidden_dim]
        memory = encoder_outputs.transpose(0, 1)
        tgt = tgt_embedding.transpose(0, 1)
        
        # 修正 key_padding_mask 的形状        
        # tgt_key_padding_mask 和 memory_key_padding_mask 用于指示哪些 token 是填充的
        # 形状为 [batch_size, seq_len]
        tgt_key_padding_mask = (target_mask == 0)
        memory_key_padding_mask = (source_mask == 0)
        
        tgt_seq_len = tgt.size(0)
        tgt_mask = self.generate_square_subsequent_mask(tgt_seq_len)

        # 确保遮蔽矩阵在正确的设备上
        tgt_mask = tgt_mask.to(device)
        decoder_output = self.transformer_decoder(
            tgt,
            memory,
            tgt_mask=tgt_mask,
            memory_mask=None,
            tgt_key_padding_mask=tgt_key_padding_mask,
            memory_key_padding_mask=memory_key_padding_mask
        )
        # torch.Size([32, 128, 768])
        decoder_output = decoder_output.transpose(0, 1)
        # torch.Size([4096, 768])
        decoder_output = decoder_output.reshape(-1, decoder_output.size(-1))
        output = self.linear(decoder_output)
        
        return output

class Modern2Ancient(nn.Module):
    def __init__(self):
        super(Modern2Ancient, self).__init__()
        # 加载保存的分词器和模型
        save_directory = './bert_ancient'
        self.tokenizer = AutoTokenizer.from_pretrained(save_directory)
        bert_model = AutoModel.from_pretrained(save_directory)

        # 冻结BERT模型参数
        for param in bert_model.parameters():
            param.requires_grad = False
            
        self.model = TransformerTranslator(bert_model, self.tokenizer.vocab_size).to(device)
        self.criterion = nn.CrossEntropyLoss(ignore_index=self.tokenizer.pad_token_id)
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=5e-5)

        self.load_checkpoint(checkpointpath)
    # 读入数据
    def load_data(self):
        source_sentences = []
        target_sentences = []
        
        with open('data/source_100000.txt', 'r', encoding='utf-8') as src_file, \
                open('data/target_100000.txt', 'r', encoding='utf-8') as tgt_file:
            source_sentences.extend(src_file.readlines())
            target_sentences.extend(tgt_file.readlines())

        source_sentences=[sentence.strip() for sentence in source_sentences]
        target_sentences=[sentence.strip() for sentence in target_sentences]
        
        return source_sentences, target_sentences
    # 过滤数据
    def filterd_data(self, seqsize=128):
        """
        过滤数据函数 根据句子长度过滤源句子和目标句子 并返回过滤后的句子对列表和数据长度。

        参数：
        source_sentences (list): 源句子列表
        target_sentences (list): 目标句子列表
        seqsize (int): 句子长度阈值 默认为128

        返回：
        tuple: 包含过滤后的源句子列表、目标句子列表和词汇长度的元组。
        """
        source_sentences, target_sentences = self.load_data()
        assert len(source_sentences) == len(target_sentences), "source_sentences和target_sentences的长度不相同!"

        # 初始化过滤后的句子对列表
        filtered_sources = []
        filtered_targets = []

        for source, target in zip(source_sentences, target_sentences):
            # 检查源句子和目标句子的长度是否都不超过seqsize
            if len(source) <= seqsize and len(target) <= seqsize:
                # 如果都满足条件，将这对句子添加到filtered_pairs列表中
                filtered_sources.append(source)
                filtered_targets.append(target)

        assert len(filtered_sources) == len(filtered_targets), "source_sentences和target_sentences的长度不相同!"
        vocab_len = len(filtered_sources)
        return filtered_sources, filtered_targets, vocab_len
    # 生成遮蔽矩阵
    def generate_square_subsequent_mask(self, sz):
        mask = torch.triu(torch.ones(sz, sz), diagonal=1).bool()
        return mask
    # 获取数据加载器
    def get_data_loader(self):

        filtered_sources, filtered_targets, vocab_len = self.filterd_data(seqsize=128)
        train_dataset = TranslationDataset(filtered_sources, filtered_targets, self.tokenizer, max_length=128)
        train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
        return train_loader
    # 保存检查点
    def save_checkpoint(self, epoch, loss, file_path="checkpoint.pth"):
        checkpoint = {
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'epoch': epoch,
            'loss': loss
        }
        torch.save(checkpoint, file_path)
        print(f"Checkpoint saved to {file_path}")
    # top-k 抽样
    def top_k_sampling(self, logits, k=50):
        # 取 logits 中概率最高的 k 个
        top_k_logits, top_k_indices = torch.topk(logits, k)
        # 对 top-k 的 logits 进行 softmax 归一化
        top_k_probs = torch.softmax(top_k_logits, dim=-1)
        # 从 top-k 的分布中抽样
        next_token = torch.multinomial(top_k_probs, 1)
        return top_k_indices[next_token.item()]
    # top-p 抽样
    def top_p_sampling(self, logits, p=0.9):
        # 对 logits 进行 softmax 归一化
        sorted_logits, sorted_indices = torch.sort(logits, descending=True)
        cumulative_probs = torch.cumsum(torch.softmax(sorted_logits, dim=-1), dim=-1)
        # 找到累积概率超过 p 的位置
        sorted_indices_to_keep = cumulative_probs <= p
        sorted_indices_to_keep[..., 1:] = sorted_indices_to_keep[..., :-1].clone()
        sorted_indices_to_keep[..., 0] = 1

        # 仅保留前 p 概率的词
        top_p_logits = sorted_logits[sorted_indices_to_keep]
        top_p_probs = torch.softmax(top_p_logits, dim=-1)
        next_token = torch.multinomial(top_p_probs, 1)
        return sorted_indices[sorted_indices_to_keep][next_token.item()]
    # 应用温度调节
    def apply_temperature(self, logits, temperature=1.0):
        return logits / temperature
    # 翻译句子
    def translate_sentence(self, sentence, max_length=128, k=50, p=0.9, temperature=1.0):
        self.model.eval()

        source_encoded = self.tokenizer(sentence, return_tensors='pt', padding='max_length',
                                truncation=True, max_length=max_length).to(device)
        source_ids = source_encoded['input_ids']
        source_mask = source_encoded['attention_mask']
        target_ids = torch.tensor([[self.tokenizer.cls_token_id]]).to(device)

        with torch.no_grad():
            for _ in range(max_length):
                target_mask = torch.ones(target_ids.shape).to(device)
                output = self.model(source_ids, source_mask, target_ids, target_mask=target_mask)
                logits = output[-1, :]

                # 应用温度调节
                logits = self.apply_temperature(logits, temperature)

                # 使用 top-k 抽样
                # next_token = self.top_k_sampling(logits, k=k)

                # 使用 top-p 抽样（可选）
                next_token = self.top_p_sampling(logits, p=p)

                next_token_id = next_token.unsqueeze(0).unsqueeze(0)
                target_ids = torch.cat([target_ids, next_token_id], dim=-1)
                
                if next_token_id.item() == self.tokenizer.sep_token_id:
                    break

        translated_sentence = self.tokenizer.decode(target_ids.squeeze(), skip_special_tokens=True)
        return translated_sentence
    # 训练模型
    def train(self, num_epochs=10, start_epoch=0):
        dataloader = self.get_data_loader()

        self.model.train()
        for epoch in range(num_epochs):
            total_loss = 0
            for batch in tqdm(dataloader, desc=f'Epoch {epoch + 1}/{num_epochs}', unit='batch'):
                source_ids, source_mask, target_ids, target_mask = [x.to(device) for x in batch]
                self.optimizer.zero_grad()
                output = self.model(source_ids, source_mask, target_ids, target_mask)
                # output = output.view(-1, output.size(-1))
                target_ids = target_ids.view(-1)
                target_ids = torch.cat((target_ids[1:], torch.tensor([0]).to(device)))
                # print(output.shape)
                # print(target_ids.shape)
                loss = self.criterion(output, target_ids)
                loss.backward()
                self.optimizer.step()
                total_loss += loss.item()
            
            avg_loss = total_loss / len(dataloader)

            true_epoch = epoch + 1 + start_epoch
            with open('m2a_loss_data.txt', 'a', encoding='UTF-8') as file:
                file.write(f'Epoch {true_epoch}, Loss: {avg_loss}\n')
            print(f'Epoch {true_epoch}, Loss: {avg_loss}\n')
            if (epoch + 1) % 5 == 0:
                self.save_checkpoint(true_epoch, avg_loss, 
                file_path=f"m2a_checkpoint{true_epoch}.pth")
                
            sentence = "故意露出一些破绽，以引诱敌人深入我方，乘机切断他的后援和前应，最终陷他于死地。"
            result = self.translate_sentence(sentence)
            print(result)
    # 加载检查点
    def load_checkpoint(self, file_path="m2a_checkpoint.pth"):
        checkpoint = torch.load(file_path)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        # epoch = checkpoint['epoch']
        # loss = checkpoint['loss']
        print(f"文言文翻译器已加载....{file_path}")
        # return model, optimizer, epoch, loss
    # 训练单个句子对
    def train_single_pair(self, source_text, target_text, lr=0.0005, num_epochs=1):
        """
        接受单个源文本和目标文本进行训练
        """
        self.model.train()

        # 优化器使用随机梯度下降
        optimizer = SGD(self.model.parameters(), lr=lr)

        source_encoded = self.tokenizer(source_text, return_tensors='pt', padding='max_length',
                                        truncation=True, max_length=128).to(device)
        target_encoded = self.tokenizer(target_text, return_tensors='pt', padding='max_length',
                                        truncation=True, max_length=128).to(device)

        source_ids = source_encoded['input_ids']
        source_mask = source_encoded['attention_mask']
        target_ids = target_encoded['input_ids']
        target_mask = target_encoded['attention_mask']

        for epoch in range(num_epochs):
            optimizer.zero_grad()
            output = self.model(source_ids, source_mask, target_ids, target_mask)
            target_ids = target_ids.view(-1)
            target_ids = torch.cat((target_ids[1:], torch.tensor([0]).to(device)))
            loss = self.criterion(output, target_ids)
            loss.backward()
            optimizer.step()

            print(f'Epoch {epoch + 1}/{num_epochs}, Loss: {loss.item()}')

# 与前端交互接口          
m2a = Modern2Ancient()
def predictM2A(sentence):
    result = m2a.translate_sentence(sentence)
    return result
def learningM2A(source,target):
    m2a.train_single_pair(source, target, 0.0008, num_epochs=1)
    print('<<<<<<<M2A模型已做适应性优化<<<<<<<<')
    
if __name__ == "__main__":
    m2a = Modern2Ancient()
    sentence = "自从文字被发明以来，能够了解的上古人物，就是经传中所提到的。唐、虞之前，帝王有谥号，但辅佐他们的大臣却没有记载。"
    result = m2a.translate_sentence(sentence)
    print(result)
    m2a.train_single_pair(sentence, "自书契之作，先民可得而闻者，经传所称，唐、虞以上，帝王有号谥。", 0.005, num_epochs=1)
    print("训练完成！")
    result = m2a.translate_sentence(sentence)
    print(result)