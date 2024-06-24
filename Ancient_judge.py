import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader, random_split
import random
from transformers import AutoTokenizer, AutoModel
from tqdm import tqdm

# 设置设备
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

class SentenceDataset(Dataset):
    def __init__(self, sentences, tokenizer, max_length=128):
        self.sentences = sentences
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.sentences)

    def __getitem__(self, idx):
        sentence, label = self.sentences[idx]
        encoded = self.tokenizer(sentence, return_tensors='pt', padding='max_length', truncation=True, max_length=self.max_length)
        input_ids = encoded['input_ids'].squeeze(0)
        attention_mask = encoded['attention_mask'].squeeze(0)
        return input_ids, attention_mask, torch.tensor(label, dtype=torch.long)

class Attention(nn.Module):
    def __init__(self, hidden_dim):
        super(Attention, self).__init__()
        self.attention = nn.Linear(hidden_dim, 1)

    def forward(self, lstm_output):
        attn_weights = torch.tanh(self.attention(lstm_output)).squeeze(-1)
        attn_weights = torch.softmax(attn_weights, dim=1)
        context = torch.bmm(attn_weights.unsqueeze(1), lstm_output).squeeze(1)
        return context, attn_weights

class LSTMAttentionClassifier(nn.Module):
    def __init__(self, bert_model, hidden_dim, output_dim):
        super(LSTMAttentionClassifier, self).__init__()
        self.bert_embedding = bert_model.embeddings
        self.lstm = nn.LSTM(bert_model.config.hidden_size, hidden_dim, batch_first=True, bidirectional=True)
        self.attention = Attention(hidden_dim * 2)
        self.fc = nn.Linear(hidden_dim * 2, output_dim)

    def forward(self, input_ids, attention_mask):
        with torch.no_grad():
            embeddings = self.bert_embedding(input_ids)
        lstm_output, _ = self.lstm(embeddings)
        context, attn_weights = self.attention(lstm_output)
        output = self.fc(context)
        return output, attn_weights
    

class Ancient_judge(nn.Module):
    def __init__(self):
        super(Ancient_judge, self).__init__()
        # 加载保存的分词器和模型
        save_directory = './bert_ancient'
        self.tokenizer = AutoTokenizer.from_pretrained(save_directory)

        # 加载模型
        bert_model = AutoModel.from_pretrained(save_directory)
        self.model = LSTMAttentionClassifier(bert_model, hidden_dim=256, output_dim=2).to(device)
        self.criterion = nn.CrossEntropyLoss()
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=1e-5)

        self.load_checkpoint('../checkpoint/judge_checkpoint10.pth')
    # 数据加载
    def load_data(self, file_path, label):
        sentences = []
        with open(file_path, 'r', encoding='utf-8') as file:
            for line in file:
                sentences.append((line.strip(), label))

        return sentences

    def get_data_loader(self):
        # 合并并打乱数据
        ancient_sentences = self.load_data('data/source_100000.txt', 1)
        modern_sentences = self.load_data('data/target_100000.txt', 0)
        all_sentences = ancient_sentences + modern_sentences
        random.shuffle(all_sentences)

        dataset = SentenceDataset(all_sentences, self.tokenizer)
        # 划分训练集和验证集
        train_size = int(0.8 * len(dataset))
        val_size = len(dataset) - train_size
        train_dataset, val_dataset = random_split(dataset, [train_size, val_size])
        train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)
        return train_loader, val_loader
    
    def save_checkpoint(self, epoch, loss, accuracy, 
                        file_path="judge_checkpoint.pth"):
        checkpoint = {
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'epoch': epoch,
            'loss': loss,
            'accuracy': accuracy
        }
        torch.save(checkpoint, file_path)
        print(f"Checkpoint saved to {file_path}")

    def load_checkpoint(self, file_path="judge_checkpoint.pth"):
        checkpoint = torch.load(file_path)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        print(f"古诗文判别器已加载...{file_path}")
        # return epoch, loss, accuracy

    def evaluate(self, data_loader):
        self.model.eval()
        correct = 0
        total = 0
        with torch.no_grad():
            for input_ids, attention_mask, labels in data_loader:
                input_ids = input_ids.to(device)
                attention_mask = attention_mask.to(device)
                labels = labels.to(device)

                outputs, _ = self.model(input_ids, attention_mask)
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()

        accuracy = correct / total
        return accuracy

    def train(self, num_epochs=3, start_epoch=0):
        
        train_loader, val_loader = self.get_data_loader()
        for epoch in range(num_epochs):
            self.model.train()
            total_loss = 0
            progress_bar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{num_epochs}")
            for input_ids, attention_mask, labels in progress_bar:
                input_ids = input_ids.to(device)
                attention_mask = attention_mask.to(device)
                labels = labels.to(device)

                self.optimizer.zero_grad()
                outputs, _ = self.model(input_ids, attention_mask)
                loss = self.criterion(outputs, labels)
                loss.backward()
                self.optimizer.step()

                total_loss += loss.item()
                progress_bar.set_postfix(loss=loss.item())

            avg_loss = total_loss / len(train_loader)
            print(f'Epoch {epoch + 1}, Loss: {avg_loss}')
            
            val_accuracy = self.evaluate(val_loader)
            print(f'Epoch {epoch + 1}, Validation Accuracy: {val_accuracy}')

            torch.cuda.empty_cache()  # Release GPU memory

            true_epoch = epoch + 1 + start_epoch
            with open('judge_loss_data.txt', 'a', encoding='UTF-8') as file:
                file.write(f'Epoch {true_epoch}, Loss: {avg_loss}, Validation Accuracy: {val_accuracy}\n')

            self.save_checkpoint(true_epoch, avg_loss, val_accuracy, file_path=f"judge_checkpoint{true_epoch}.pth")

    def predict_sentence(self, sentence):
        self.model.eval()
        encoded = self.tokenizer(sentence, return_tensors='pt', padding='max_length', truncation=True, max_length=128)
        input_ids = encoded['input_ids'].to(device)
        attention_mask = encoded['attention_mask'].to(device)

        with torch.no_grad():
            outputs, _ = self.model(input_ids, attention_mask)
            _, predicted = torch.max(outputs, 1)
        
        return "古文" if predicted.item() == 0 else "现代文"

if __name__ == '__main__':
    an_judge = Ancient_judge()
    print(an_judge.predict_sentence("我爱中国"))


#     an_judge.train(num_epochs=5, start_epoch=10)
#     print(an_judge.predict_sentence("我爱中国"))