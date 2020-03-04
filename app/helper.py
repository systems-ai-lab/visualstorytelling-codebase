import os
import torch
import time
import ntpath
from torch.utils.tensorboard import SummaryWriter

class Vocabulary(object):
    """This class is a custom data structure object with the basic functional
    such as adding new word, retrieving the index of word, and find the length
    of the 
    """
    def __init__(self):
        self.word2idx = {}
        self.idx2word = {}
        self.idx = 0

    def add_word(self, word):
        if not word in self.word2idx:
            self.word2idx[word] = self.idx
            self.idx2word[self.idx] = word
            self.idx += 1

    def __call__(self, word):
        if not word in self.word2idx:
            return self.word2idx['<unk>']
        return self.word2idx[word]

    def __len__(self):
        return len(self.word2idx)

class Logger():
    def __init__(self, model_dir, tensorboard_dir, file_config):
        self.model_dir = model_dir
        self.tensorboard_dir = tensorboard_dir
        
        # check the directory for save the model is exist or not
        if not os.path.exists(model_dir):
            os.mkdir(model_dir)
        
        if not os.path.exists(tensorboard_dir):
            os.mkdir(tensorboard_dir)
        
        # experiment identifier
        experiment_id = ntpath.basename(file_config)
        self.experiment_id = os.path.splitext(experiment_id)[0]

        # check and create directory for save the models
        if not os.path.exists(os.path.join(self.model_dir, self.experiment_id)):
            os.mkdir(os.path.join(self.model_dir, self.experiment_id))
        
        # check the model directory is empty or not
        if os.listdir(os.path.join(self.model_dir, self.experiment_id)) :
            raise RuntimeError('The model directory is not empty')
        
        # check and create directory for save the tensorboard
        if not os.path.exists(os.path.join(self.tensorboard_dir, self.experiment_id)):
            os.mkdir(os.path.join(self.tensorboard_dir, self.experiment_id))
        
        # check the tensorboard directory is empty or not
        if os.listdir(os.path.join(self.tensorboard_dir, self.experiment_id)) :
            raise RuntimeError('The tensorboard directory is not empty')
        
    def save_model(self, model_encoder, model_decoder, epoch):
        current_time = time.strftime("%Y%m%d_%H%M%S")
        torch.save(model_encoder.state_dict(), os.path.join(self.model_dir, self.experiment_id, '%s-encoder-%d.pkl' %(current_time, epoch+1)))
        torch.save(model_decoder.state_dict(), os.path.join(self.model_dir, self.experiment_id, '%s-decoder-%d.pkl' %(current_time, epoch+1)))
    
    def tensorboard_scalar(self, tag, scalar_value, global_step):
        # create object for tensorboard writing
        writer = SummaryWriter(os.path.join(self.tensorboard_dir, self.experiment_id, tag))
        writer.add_scalar(tag=self.experiment_id+"_"+tag, scalar_value=scalar_value, global_step=global_step)
    
    def tensorboard_scalars(self, tag, scalar_list, global_step):
        obj = {}
        for id, data in enumerate(scalar_list):
            obj[data[0]] = data[1]
        
        writer = SummaryWriter(os.path.join(self.tensorboard_dir, self.experiment_id, tag))
        writer.add_scalars(self.experiment_id+"_"+tag, obj, global_step)

    def tensorboard_graph(self, model, input_to_model):
        writer = SummaryWriter(os.path.join(self.tensorboard_dir, self.experiment_id))
        writer.add_graph(model=model, input_to_model=input_to_model)
        writer.close()



