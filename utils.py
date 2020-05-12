import os, torch, shutil, json, dill, pickle

class Utils:
    @staticmethod
    def save_checkpoint(state, is_best, checkpoint, epoch):
        """
        모델을 저장합니다.
        베스트 모델인 경우 카피본을 하나 더 만듭니다.
        """
        filepath = os.path.join(checkpoint, 'e{:02d}.pth.tar'.format(epoch))
        if not os.path.exists(checkpoint):
            print("Checkpoint Directory does not exist! Making directory {}".format(checkpoint))
            os.makedirs(checkpoint)
        else:
            print("Checkpoint Directory exists!")
        torch.save(state, filepath)
        if is_best:
            shutil.copyfile(filepath, os.path.join(checkpoint, 'best.pth.tar'))

    @staticmethod
    def load_checkpoint(checkpoint, model, optimizer=None):
        """
        저장된 모델 파라미터를 불러와 업데이트합니다.
        """
        print(checkpoint)
        if not os.path.exists(checkpoint):
            
            print("File doesn't exist {}".format(checkpoint))
            raise Exception
        
        checkpoint = torch.load(checkpoint, map_location=torch.device('cpu'))
        model.load_state_dict(checkpoint['state_dict'])

        if optimizer:
            optimizer.load_state_dict(checkpoint['optim_dict'])
            print("OPTIMIZER LOAD THE STATE DICT")

        #return checkpoint
        return model
    
    @staticmethod
    def save_dict_to_json(d, json_path):
        """
        dictionary를 json 파일로 저장한다.
        """
        with open(json_path, 'w') as f:
            # We need to convert the values to float for json (it doesn't accept np.array, np.float, )
            d = {k: float(v) for k, v in d.items()}
            json.dump(d, f, indent=4)
    
    @staticmethod
    def save_field(data, filename):
        with open(os.path.dirname(os.path.abspath(__file__)) + "/field" + "/{}.field".format(filename), 'wb') as t:
            dill.dump(data, t)
            print("FIELD SAVE DONE")
    
    @staticmethod
    def load_field(filename):
        with open(os.path.dirname(os.path.abspath(__file__)) + "/field" + "/{}.field".format(filename), 'rb') as t:
            data = dill.load(t)
            print("FIELD LOAD DONE")
            return data
    
    @staticmethod
    def load_pickle(filename):
        with open(os.path.dirname(os.path.abspath(__file__)) + "/pickle" + "/{}.pickle".format(filename), 'rb') as f:
            data = pickle.load(f) # 단 한줄씩 읽어옴
        return data

if __name__ == "__main__":
    print(os.path.dirname(os.path.abspath(__file__)) + "/pickle" + "/{}.pickle".format('dataloader'))
