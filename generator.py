# -*- conding: utf-8 -*-
from utils import Utils
import torch, os, random
from collections import OrderedDict
import torch.nn.functional as F
from train import Net
from tqdm import tqdm, trange
import torch.optim as optim
DEVICE = "cuda:0" if torch.cuda.is_available() else "cpu" # gpu 사용불가시 cpu로 학습 설정

class Generator:
    def __init__(self):
    # def __init__(self, is_random=True, category=None, ingredients=None, init_char=None):

        # torchtext 필드 지정
        self.product_field = Utils.load_field('product')
        self.category_field = Utils.load_field('category')
        self.ingredients_field = Utils.load_field('ingredients')

        # 모델 초기화
        self.model_initialize()

        # optimizer 세팅
        self.optimizer = optim.Adam(self.Model.parameters(), lr=1e-3, weight_decay=1e-3)

        # best_model 경로
        self.best_model_path = os.path.dirname(os.path.abspath(__file__)) + "/model" + "/best.pth.tar"
        # "model/e50.pth.tar" #"model/best.pth.tar"

        # if self.is_random is not True:
        #     self.category = category
        self.init_char_list = Utils.load_pickle('init_char_list')
        self.fine_char_list = self.get_fine_char_list()
        self.wired_char_list = self.get_wired_char_list()

    def get_fine_char_list(self):
        """
        [
            (아,24134),
            (이,14134),
            (맨,12312)
        ]
        """
        
        len_list = len(self.init_char_list)
        top_ = int(len_list * 0.3)

        fine_char_list = list()
        for n, char in enumerate(self.init_char_list):
            fine_char_list.append(char[0])
            if n == top_:
                break
        return fine_char_list
    
    def get_wired_char_list(self):
        #reversed_ = self.init
        len_list = len(self.init_char_list)
        top_ = int(len_list * 0.3)

        wired_char_list = list()
        for n in range(top_):
            wired_char_list.append(self.init_char_list[n * (-1)-1])
        return wired_char_list


    def set_all_inputs(self, category, ingredients_list, init_char):
        self.category = category
        self.init_char = init_char
        self.ingredients_list = ingredients_list
        self.generated_name = []
        self.generated_name.append(self.init_char)


    def initialize_all_inputs(self):
        # 카테고리
        self.category = None
        # 최초 시작 문자열
        self.init_char = None
        # 성분리스트
        self.ingredients_list = None
        # 생성된 문자열 
        self.generated_name = []
    
    def initialize_category_ingredients(self):
        # 카테고리
        self.category = None
        # 성분리스트
        self.ingredients_list = None


    def randomize_all_inputs(self):
        # 카테고리
        self.category = self.get_random_category()
        # 최초 시작 문자열
        self.init_char = self.get_random_char()
        # 성분리스트
        self.ingredients_list = self.get_random_igd()
        self.ingredients_list.append('정제수')
        # 생성된 문자열
        self.generated_name = [self.init_char]
    
    def randomize_partial_inputs(self, category=None, ingredients_list=None, init_char=None, mode='fine'):
        if category is not None:
            self.category = category
        else:
            self.category = self.get_random_category()
        
        if ingredients_list is not None:
            self.ingredients_list = ingredients_list
            self.ingredients_list.append('정제수')
        else:
            self.ingredients_list = self.get_random_igd()
            self.ingredients_list.append('정제수')
        
        if init_char is not None:
            self.init_char = init_char
        else:
            if mode == 'fine':
            #self.init_char = self.get_random_char()
                self.init_char = self.get_fine_char()
            elif mode == 'wired':
                self.init_char = self.get_wired_char()
        # 생성된 문자열
        self.generated_name = [self.init_char]
        # print("CATEGORY : {}".format(self.category))
        # print("INGREDIENTS : {}".format(self.ingredients_list))
        # print("INIT_CHAR : {}".format(self.init_char))
    
    def model_initialize(self):
        self.Model = Net(vocab_size=len(self.product_field.vocab.stoi), 
                            embedding_dim=256,
                            nb_category=len(self.category_field.vocab.stoi)-1,
                            nb_ingredients=len(self.ingredients_field.vocab.stoi)-1,
                            lstm_nb_layers=3, 
                            lstm_hidden_dim=256, 
                            fc_out=128, 
                            dropout_p=0.5, ).to(DEVICE)
        print("Model Initialized")
    
    def load_best_model(self):
        self.Model = Utils.load_checkpoint(self.best_model_path, self.Model, self.optimizer)
        print("LOAD BEST MODEL")
    
    def get_random_category(self):
        box = list(self.category_field.vocab.stoi.keys())
        selected = random.choice(box)
        return selected

    # 랜덤 성분 선택
    def get_random_igd(self):
        box = list(self.ingredients_field.vocab.stoi.keys())
        selected = random.sample(box, random.randrange(20,50))
        return selected
    
    # 랜덤 시작 문자열 선택
    def get_random_char(self):
        box = list(self.product_field.vocab.stoi.keys())
        selected = random.choice(box)
        return selected
    
    def get_fine_char(self):
        selected = random.choice(self.fine_char_list)
        return selected
    
    def get_wired_char(self):
        selected = random.choice(self.wired_char_list)
        return selected
    
    def get_random_category(self):
        box = list(self.category_field.vocab.stoi.keys())
        selected = random.choice(box)
        return selected

    def generate_next_char(self, string):
        """
        카테고리, 성분들을 받아 다음글자들의 확률을 리턴
        ingredients_list = ['정제수','쌀추출물']
        category = '스킨케어'

        """
        #print(self.category)
        category_tensor = self.category_field.process([self.category]).float().to(DEVICE)
        # print("갸 : ",self.ingredients_list)
        ingredients_tensor = self.ingredients_field.process([self.ingredients_list]).float().to(DEVICE)

        #self.init_char = self.init_char.lower()
        # 한글자 잘라줘야 함
        self.product_field.fix_length = None
        first_char_tensor = self.product_field.process([string.upper()])[:, :-1].to(DEVICE)
        bsz, first_char_tensor_length = first_char_tensor.size()

        # 인풋을 모델에 넣어 출력합니다.
        #print(self.Model)
        self.Model.eval()
        with torch.no_grad():
            # batch_size = 1
            hidden = self.Model.init_hidden(1)

            for step in range(first_char_tensor_length):
                with torch.no_grad():
                    outputs, hidden = self.Model(category_tensor, ingredients_tensor, first_char_tensor[:, step], hidden)
                    # print("step : ", step)
                    # print("outputs : ", outputs.size())
                probabilities = F.softmax(outputs, 1)

        return probabilities.squeeze()

    def generate_name(self):
        #chr_list = [self.init_char]
        #print("initial_prime: ", self.generated_name)

        while self.generated_name[-1] != "<eos>":
            #print("생성된 문자열 : {}".format(self.generated_name))
            current_str = "".join(self.generated_name)
            #print("INPUT_CHAR : ", current_str)
            probabilities = self.generate_next_char(current_str)
            max_idx = probabilities.argmax().item()
            next_chr = self.product_field.vocab.itos[max_idx].replace("<pad>"," ")
            
            #print("OUTPUT_CHAR : ", next_chr)
            self.generated_name.append(next_chr)
            #print("생성된 문자열 : ", self.generated_name)

        #print(self.generated_name)
        generated_name = "".join(self.generated_name[:-1])
        #print("generated_name: ", generated_name)
        return generated_name

    def clean_beam_basket(self, basket, beam_width):
        """
        가장 확률이 높은 beam_width개만 남기고 바스켓을 비운다.
        """
        _tmp_basket = basket.copy()
        to_remove = sorted(_tmp_basket.items(), key=lambda x: x[1], reverse=True)[beam_width:]
        for item in to_remove:
            _tmp_basket.pop(item[0])

        return _tmp_basket

    def beam_search(self, beam_width, init_char, category, ingredients, alpha=0.7):

        beam_basket = OrderedDict()
        beam_basket[init_char] = 0.0
        counter = 0
        
        while True:
            counter += 1

            # 바스켓을 청소합니다.
            beam_basket = self.clean_beam_basket(beam_basket, beam_width)

            # 만약 바스켓에 모든 아이템이 <eos>가 있으면 루프를 멈춘다.
            eos_cnt = 0
            print("eos_cnt : ", eos_cnt)
            for k in beam_basket.keys():
                if "<eos>" in k:
                    eos_cnt += 1
            if eos_cnt == beam_width:
                # print("all items have <eos>")
                break

            # 모든 key를 돌면서
            ## <eos>가 없는 경우 inference한다.
            new_entries = {}
            to_remove = []
            for k in beam_basket.keys():
                if "<eos>" not in k:
                    probabilities = self.generate_next_char(init_char)
                    for ix, prob in enumerate(probabilities):
                        new_k = k + self.product_field.vocab.itos[ix]
                        added_probability = beam_basket[k] + torch.log(prob).item()
                        len_k = len(k.replace("<eos>", ""))
                        normalized_probability = (1 / (len(k) ** alpha)) * added_probability
                        new_entries[new_k] = normalized_probability
                    to_remove.append(k)
        
            # 그리고 기존 key를 beam_basket에서 지운다.
            for k in to_remove:
                beam_basket.pop(k)

            # 새로운 키를 바스켓에 채워넣는다.
            for k, v in new_entries.items():
                beam_basket[k] = v

        final_list = []
        print(final_list)
        for k, v in beam_basket.items():
            refined_k = k.replace("<eos>", "").capitalize()
            final_list.append(refined_k)
            final_prob = np.exp(v)

        return final_list


def main(category=None, ingredients_list=None, init_char=None, N=10, debug=True):
    generator = Generator()
    
    #print("2: " , generator.)
    # 베스트 모델 불러오기
    generator.load_best_model()

    # N개의 랜덤 이름 생성
    name_list = list()
    
    # 모든 인풋값 초기화
    generator.initialize_all_inputs()
    
    for _ in trange(N):
        generator.initialize_category_ingredients()
        if category is not None and ingredients_list is not None and init_char is not None:
            print("빔서치")
            generator.set_all_inputs(category, ingredients_list, init_char)
            if debug: 
                print("제품 카테고리 : " , generator.init_char)
                print("제품 카테고리 : " , generator.category)
                print("사용된 성분 : " , generator.ingredients_list)
            # name_list = generator.beam_search(N, init_char, category, ingredients_list)
            # break
        
        elif category or ingredients_list or init_char:
            #print("Not all None")
            generator.randomize_partial_inputs(category, ingredients_list, init_char)
            if debug:
                print("제품 카테고리 : " , generator.init_char)
                print("제품 카테고리 : " , generator.category)
                print("사용된 성분 : " , generator.ingredients_list)

        elif category is None and ingredients_list is None and init_char is None:
            # 전체 랜덤 인풋값 가져오기
            generator.randomize_all_inputs()
            if debug:
                print("제품 카테고리 : " , generator.init_char)
                print("제품 카테고리 : " , generator.category)
                print("사용된 성분 : " , generator.ingredients_list)
        
        name_list.append(generator.generate_name())
    
    # print("CATEGORY : {}".format())
    # print("INGREDINE : {}".format())
    # print("CATEGORY : {}".format())

    return name_list


if __name__ == "__main__":

    
    # 한계 & 보완

    # 성분간의 상관관계를 모른다 -> 어떤 성분을 인풋으로 넣어야할지... e.g) 정제수는 꼭들어가야함, 글리세린 거의 들어감 등
    # 시작 글자를 정해줘야 한다 -> 어떠한 컨셉을 인풋으로 줄 수 있을까? / 시작글자 통계를 통해 토큰 범위 지정

    # 주요 성분만 넣고, 그걸 통해 예측해서 성분 조합을 만들고 그것을 인풋으로 사용!

    category = '스킨케어'
    init_char = "클린"
    ing = ['녹차',
            '블랙베리체리나무',
            '쑥',
            '녹차수',
            '히알루론산합성효소',
            '히포파에람노이데스프룻추출물',
            '힙네아 무스키포르미스추출물',
            '힙시지구스 울마리우스균사체추출물',
            '녹차오일',
            '낙화생유',
            '녹차엑스',
            '소듐솝베이스']

    name_list = main(
        # 빔서치
        # category=category,
        # init_char=init_char,
        # ingredients_list = ing
        # 랜덤
        category='스킨케어', # 스킨케어
        init_char=None,
        ingredients_list = None
    )
    print(name_list)