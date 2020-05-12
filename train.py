import torch, pandas as pd
from torchtext import data
import torch.nn as nn
#import torch.nn.functional as F
import torch.optim as optim
import os, logging, sys#, shutil, json, pickle
from sklearn.metrics import accuracy_score
from pprint import pprint
from tqdm import tqdm, trange
from utils import Utils
#from confidential__ import RawDataLoader, query
DEVICE = "cuda:0" if torch.cuda.is_available() else "cpu" # gpu 사용불가시 cpu로 학습 설정
sys.path.append(os.path.dirname(os.path.abspath(os.path.dirname(__file__))))

DEBUG=False


class Processor:
    DEBUG = False
    @staticmethod
    def tokenizer(text):
        return list(text.upper())
    
    @staticmethod
    def single_onehot(alist, vocab):
        # if DEBUG: print("single_onehot")
        # if DEBUG: print('alist :\n', alist)
        
        # alist = [2, 1, 1]
        # _tensor = [[1, 0, 0]]
        _tensor = torch.tensor(alist).data.sub_(1).unsqueeze(1)
        # if DEBUG: print("_tensor : \n",_tensor)

        # _onehot = [[0, 0], [0, 0], [0, 0]]
        _onehot = torch.zeros((len(alist), len(vocab) - 1), dtype=torch.float)
        

        # _onehot = [[0, 1], [1, 0], [1, 0]]
        _onehot.scatter_(1, _tensor, 1)
        # if DEBUG: print("_onehot : \n",_onehot)
        return _onehot
    
    @staticmethod
    def spliter(text):
        return text.split(";")

    @staticmethod
    def multiple_onehot(alist, vocab):
        if DEBUG: print("multiple_onehot")
        if DEBUG: print('alist :\n', alist)
        
        _tensor = torch.tensor(alist).data.sub_(1)
        #print("후후", _tensor)
        # if DEBUG: print("_tensor : \n",_tensor)
        '''
        _tensor
        [[   1,    8,   44,  ...,    0,    0,    0],
        [   1,    7,   15,  ...,    0,    0,    0],
        [   1, 1210, 1249,  ...,    0,    0,    0]]
        '''
        
        _onehot = torch.zeros((len(alist), len(vocab) - 1), dtype=torch.float)
        #print("후후", _onehot)
        # _onehot = [[0, 1], [1, 0], [1, 0]]
        
        _onehot.scatter_(1, _tensor, 1)
        # if DEBUG: print("_onehot : \n",_onehot)
        return _onehot
    
    @staticmethod
    def preprocess_igd(ingredients_list):
        '''
        ingredients_list
        ['정제수', '사이클로메티콘', '다이프로필렌글라이콜', '티타늄다이옥사이드', '다이페닐실록시페닐트라이메티콘', '피이지-10다이메티콘', '헥실라우레이트', '하이드로제네이티드폴리데센',     
        '메틸메타크릴레이트크로스폴리머', '다이메티콘', '다이메티콘/비닐다이메티콘크로스폴리머', '다이스테아다이모늄헥토라이트', '소듐클로라이드', '탈크', '트라이에톡시카프릴릴실레인', '부틸렌글라이콜', 
        '티트리잎추출물', '캐모마일꽃추출물', '로즈마리잎추출물', '다이소듐이디티에이', '메틸파라벤', '프로필파라벤', '향료', '벤질벤조에이트', '부틸페닐메틸프로피오날', '시트로넬올', '제라니올', 
        '헥실신나몰', '리모넨', '리날룰', '크로뮴옥사이드그린', '울트라마린', '황색산화철']
        '''
        if DEBUG: print("START",ingredients_list)
        return ingredients_list


class DataLoader(object):
    def __init__(self, data_dir, batch_size):
        self.data_dir = data_dir
        self.max_length = self.get_max_length()
        self.PRODUCT = data.Field(sequential=True, use_vocab=True, tokenize=Processor.tokenizer, batch_first=True, init_token="<bos>", eos_token="<eos>", fix_length=self.max_length+3)
        self.CATEGORY = data.Field(sequential=False, use_vocab=True, postprocessing=Processor.single_onehot)
        self.INGREDIENTS = data.Field(sequential=True, use_vocab=True, tokenize=Processor.spliter, preprocessing=Processor.preprocess_igd , postprocessing=Processor.multiple_onehot, batch_first=True,)
        
        self.train_ds, self.val_ds = data.TabularDataset.splits(
            path=data_dir, skip_header=True, 
            train='train.csv',
            validation='val.csv', 
            format='csv',
            fields=[(u'MainCategory', self.CATEGORY), (u'Name', self.PRODUCT), (u'Ingredients', self.INGREDIENTS)]
        )
        print("train_ds length : {}".format(len(self.train_ds)))
        print("val_ds length : {}".format(len(self.val_ds)))
        
        self.build_vocab()
        
        self.train_iter, self.val_iter = data.BucketIterator.splits(
            (self.train_ds, self.val_ds), batch_sizes=(batch_size, batch_size), device=DEVICE,
            repeat=False, 
            sort_key=lambda x: len(x.Name))



    def build_vocab(self):
        self.PRODUCT.build_vocab(self.train_ds, self.val_ds)
        self.CATEGORY.build_vocab(self.train_ds, self.val_ds)
        self.INGREDIENTS.build_vocab(self.train_ds, self.val_ds)
        print("Build Vocab Done..")
    
    def get_max_length(self):
        #df = pd.read_csv(os.path.join(self.data_dir, u"/train.csv"))
        df = pd.read_csv(self.data_dir + '/train.csv')
        
        max_length = int(max(df['Name'].str.len().values))
        return max_length


class Net(nn.Module):
    DEBUG=False
    def __init__(self, vocab_size, embedding_dim, nb_category, nb_ingredients, lstm_nb_layers, lstm_hidden_dim, fc_out, dropout_p):
        super(Net, self).__init__()

        # 제품명의 토큰 수
        self.vocab_size = vocab_size
        # 제품명의 임베딩 차원
        self.embedding_dim = embedding_dim
        # 제품 카테고리 수
        self.nb_category = nb_category
        # 제품 성분 수
        self.nb_ingredients = nb_ingredients
        # lstm 레이어수
        self.lstm_nb_layers = lstm_nb_layers
        # lstm return의 차원
        self.lstm_hidden_dim = lstm_hidden_dim
        
        #self.fc_out = fc_out
        # Dropout 비율
        self.dropout_p = dropout_p
        self.device = DEVICE #"cuda:0" if torch.cuda.is_available() else "cpu"

        # 제품명 임베딩 layer
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        # LSTM layer
        self.lstm = nn.LSTM(embedding_dim + nb_category + nb_ingredients, lstm_hidden_dim, num_layers=lstm_nb_layers, batch_first=True)
        # Linear Layer
        self.fc_inter = nn.Linear(lstm_hidden_dim, fc_out)
        # Apply dropout
        self.dropout = nn.Dropout(dropout_p)
        # Decode layer -> 제품명 토큰숫자로 아웃풋
        self.decoder = nn.Linear(fc_out, vocab_size)

    # 히든레이어 초기화
    def init_hidden(self, batch_size):
        return (torch.zeros(self.lstm_nb_layers, batch_size, self.lstm_hidden_dim).float().to(self.device),
                torch.zeros(self.lstm_nb_layers, batch_size, self.lstm_hidden_dim).float().to(self.device))

    # 
    def forward(self, category, ingredients, inputs, hidden, isDebug=False):
        if DEBUG: print("category:", category.size())
        if DEBUG: print("ingredients:", ingredients.size())
        if DEBUG: print("inputs:", inputs.size())

        # 인풋 제품명 임베딩
        embed = self.embedding(inputs)
        if DEBUG: print("embed:", embed.size())

        # 카테고리 벡터 + 성분벡터 + 제품명임베딩
        inputs_combined = torch.cat([category, ingredients, embed], dim=1)
        if DEBUG: print("inputs_combined:", inputs_combined.size())

        # 모델 In
        lstm_out, hidden = self.lstm(inputs_combined.unsqueeze(1), hidden)
        if DEBUG: print("lstm_out:", lstm_out.size())
        if DEBUG: print("last_hidden_state:", hidden[0].size())
        if DEBUG: print("last_cell_state:", hidden[1].size())

        # Linear In
        fc_inter_out = self.fc_inter(lstm_out.squeeze(1))
        if DEBUG: print("fc_inter_out:", fc_inter_out.size())

        # Dropout
        dropout_out = self.dropout(fc_inter_out)

        # Decode
        decoder_out = self.decoder(dropout_out)
        if DEBUG: print("decoder_out:", decoder_out.size())

        return decoder_out, hidden


class MetricCalculator():
    """
    loss와 accuracy를 기록하기 위한 도구입니다.
    """
    def __init__(self):
        self.accuracy = 0
        self.loss_accumulated = 0
        self.average_loss = 0
        self.updated_cnt = 0

        self.predicted_labels_holder = []
        self.actual_labels_holder = []

    def update(self, outputs, labels, loss):
        self.updated_cnt += 1

        predicted_labels = outputs.max(1)[1]
        self.predicted_labels_holder.append(predicted_labels)
        self.actual_labels_holder.append(labels)
        self.loss_accumulated += loss


    def calculate_metric(self):

        predicted_labels = torch.cat(self.predicted_labels_holder).cpu().numpy()
        actual_labels = torch.cat(self.actual_labels_holder).cpu().numpy()

        self.accuracy = accuracy_score(actual_labels, predicted_labels)
        self.average_loss = self.loss_accumulated / self.updated_cnt


    def reset(self):
        self.accuracy = 0
        self.loss_accumulated = 0
        self.average_loss = 0
        self.updated_cnt = 0

    def export(self):
        return {
            'loss': self.average_loss,
            'accuracy': self.accuracy,
        }



class Trainer:
    def __init__(self, dataloader, model, batch_size = 16, epochs = 50, learning_rate = 1e-3, learning_rateweight_decay = 1e-3):
        self.dataloader = dataloader
        self.train_data_iter = self.dataloader.train_iter
        self.val_data_iter = self.dataloader.val_iter
        self.learning_rate = learning_rate
        self.weight_decay = learning_rateweight_decay
        self.model = model
        self.model_dir = "model"
        self.num_epochs = epochs
        self.batch_size = batch_size
        self.num_steps = len(self.train_data_iter.dataset.examples) // self.batch_size + 1
        self.optimizer = optim.Adam(self.model.parameters(), lr=learning_rate, weight_decay=self.weight_decay)

    def loss_fn(self, outputs, labels):
        loss_function = nn.CrossEntropyLoss()
        loss = loss_function(outputs, labels)
        return loss

    def train(self):
        """
        training data를 사용해서 모델을 학습하는 함수.
        """
    
        metric_watcher = MetricCalculator()
        self.model.train()
        
        print("Start Training")
        for ix, batch in tqdm(enumerate(self.train_data_iter), total=len(self.train_data_iter)):
            inputs = batch.Name[:, :-1].to(DEVICE)
            targets = batch.Name[:, 1:].to(DEVICE)
            category = batch.MainCategory.float().to(DEVICE)
            ingredients = batch.Ingredients.float().to(DEVICE)

            hidden = self.model.init_hidden(inputs.size(0))

            loss = 0.0

            for step in trange(inputs.size(1)):
                outputs, hidden = self.model.forward(category, ingredients, inputs[:, step], hidden)
                current_loss = self.loss_fn(outputs, targets[:, step])

                metric_watcher.update(outputs, targets[:, step], current_loss.detach())
                #print(current_loss.data.cpu().numpy())
                #loss += current_loss.cpu()
                #loss += current_loss.data.cpu()
                #loss += current_loss.detach()
                loss += current_loss
                
                del outputs

            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()#

            del inputs
            del targets
            del category
            del ingredients
            del loss

        metric_watcher.calculate_metric()
        metrics_string = "loss: {:05.3f}, acc: {:05.3f}".format(
            metric_watcher.average_loss,
            metric_watcher.accuracy
        )
        print("- Train metrics: " + metrics_string)


    def evaluate(self):
        """
        validation data를 사용해서 학습한 모델의 성능을 평가하는 함수. 
        """
    
        metric_watcher = MetricCalculator()

        # set model to evaluation mode
        self.model.eval()

        # compute metrics over the dataset
        print("Start Evaluating")
        for ix, batch in enumerate(self.val_data_iter):
            inputs = batch.Name[:, :-1].to(DEVICE)
            targets = batch.Name[:, 1:].to(DEVICE)
            category = batch.MainCategory.float().to(DEVICE)
            ingredients = batch.Ingredients.float().to(DEVICE)

            hidden = self.model.init_hidden(inputs.size(0))

            loss = 0.0
        
            for step in trange(inputs.size(1)):
                outputs, hidden = self.model.forward(category, ingredients, inputs[:, step], hidden)
                current_loss = self.loss_fn(outputs, targets[:, step])

                metric_watcher.update(outputs, targets[:, step], current_loss.detach())
                loss += current_loss

        metric_watcher.calculate_metric()

        # compute mean of all metrics in summary
        metric_watcher.calculate_metric()
        metrics_string = "loss: {:05.3f}, acc: {:05.3f}".format(
            metric_watcher.average_loss,
            metric_watcher.accuracy
        )
        print("- Eval metrics: " + metrics_string)

        return metric_watcher.export()

    def train_and_evaluate(self, restore_file=None):
    
        """
        num_epochs만큼 학습과 검증을 하는 함수.
        """

        # if restore_file is given, load the checkpoint
        if restore_file is not None:
            restore_path = os.path.join(self.model_dir, restore_file + '.pth.tar')
            print("Restoring parameters from {}".format(restore_path))
            Utils.load_checkpoint(restore_path, model, optimizer)


        best_val_acc = 0.0

        for epoch in trange(self.num_epochs):

            print("Epoch {}/{}".format(epoch + 1, self.num_epochs))

            # compute number of batches in one epoch
            # num_steps = len(train_data_iter.dataset.examples) // batch_size + 1
            self.train()

            val_metrics = self.evaluate()
            val_acc = val_metrics['accuracy']
            is_best = val_acc > best_val_acc

            # Save weights
            Utils.save_checkpoint({'epoch': epoch+1,
                                'state_dict': self.model.state_dict(),
                                'optim_dict': self.optimizer.state_dict()},
                                is_best=is_best,
                                checkpoint=self.model_dir,
                                epoch=epoch+1)

            if is_best:
                print("-- Found new best accuracy")
                best_val_acc = val_acc

                best_json_path = os.path.join(self.model_dir, "metrics_val_best_weights.json")
                Utils.save_dict_to_json(val_metrics, best_json_path)



def main(data_dir, batch_size):
    # 데이터 로더 만들기
    dataloader = DataLoader(data_dir, batch_size)

    # 모델 파라미터
    nb_product_vocab=len(dataloader.PRODUCT.vocab.stoi)
    nb_category=len(dataloader.CATEGORY.vocab.stoi)-1
    nb_ingredients=len(dataloader.INGREDIENTS.vocab.stoi)-1

    # 모델링

    
    # 트레이너
    trainer = Trainer(dataloader, model)
    
    # 학습하기 & 평가하기
    trainer.train_and_evaluate()

if __name__ == "__main__":
    DEBUG = False
    data_dir = os.path.dirname(os.path.abspath(__file__)) + "/data"
    batch_size = 128
    print("BATCH_SIZE : {}".format(batch_size))

    # Raw데이터 불러오기
    # raw_loader = RawDataLoader(is_sample=False,
    #         query=query,
    #         path=data_dir
    #     )
    # 학습 & 평가셋  나누기
    # raw_loader.train_val_split()
    
    # CSV저장하기
    # raw_loader.save_csv()

    #학습하기
    #main(data_dir, batch_size)
    

    # 필드저장    
    # dataloader = DataLoader(data_dir, batch_size)
    # Utils.save_field(dataloader.PRODUCT, "product")
    # Utils.save_field(dataloader.INGREDIENTS, "ingredients")
    # Utils.save_field(dataloader.CATEGORY, "category")