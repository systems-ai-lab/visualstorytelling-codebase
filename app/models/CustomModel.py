import torch
import torch.nn as nn
import torchvision.models as models
import numpy as np

class Encoder(nn.Module):
    '''
        The encoding process flow is by extract the visual features and then
        pass it to the RNN, as the input is sequence of images
    '''
    def __init__(self, encoder_output_size, hidden_size, num_layers, is_bidirectional):
        super(Encoder, self).__init__()
        
        resnet = models.resnet152(pretrained=True)
        modules = list(resnet.children())[:-1]
        self.resnet = nn.Sequential(*modules)
        for param in self.resnet.parameters():
            param.requires_grad = False
        
        self.hidden_size = hidden_size
        self.is_bidirectional = is_bidirectional

        # https://pytorch.org/docs/stable/nn.html#linear
        self.linear_visual_features = nn.Linear(in_features = resnet.fc.in_features, 
                                out_features = encoder_output_size)

        # hidden size will multiplied by 2 if the bidirectional is True
        self.linear_visual_sequence = nn.Linear(in_features = \
            hidden_size*(2 if is_bidirectional else 1)+encoder_output_size, \
                out_features=hidden_size*(2 if is_bidirectional else 1))
        
        # https://pytorch.org/docs/stable/nn.html#batchnorm1d
        self.bn_visual_features = nn.BatchNorm1d(num_features = encoder_output_size, 
                                                 momentum= 0.01)

        # https://pytorch.org/docs/stable/nn.html#batchnorm1d
        self.bn_visual_sequence = nn.BatchNorm1d(num_features = hidden_size*(2 if is_bidirectional else 1),\
                                                momentum = 0.01)
        
        # https://pytorch.org/docs/stable/nn.html#dropout 
        self.dropout = nn.Dropout(p=0.5)

        # https://pytorch.org/docs/stable/nn.html#lstm                        
        self.lstm = nn.LSTM(input_size = encoder_output_size, 
                            hidden_size = hidden_size, 
                            num_layers = num_layers, 
                            batch_first=True, 
                            bidirectional = is_bidirectional, 
                            dropout=0.5) # if bidirectional is True, the output size will be 2 * hidden_size
    
    def forward(self, image_sequence):
        '''
            image_sequence dimension: batch_size, story_length, channels,
                                      width, height
        '''
        sequence_size = image_sequence.size()
        image_sequence = image_sequence.view(-1, sequence_size[2], 
                                                 sequence_size[3], 
                                                 sequence_size[4]) # removing the story_length dimension
        '''
            image_sequence dimension:   batch_size * story_length, 
                                        channel, width, height
        '''
        visual_features = self.resnet(image_sequence)
        visual_features = self.linear_visual_features(visual_features.squeeze())
        visual_features = self.bn_visual_features(visual_features)
        
        visual_features = visual_features.view(sequence_size[0], sequence_size[1], -1) # restoring back story_length dimension
        
        '''
            returning back the batch and story length dimension
            visual_features dimension:  batch_size, story_length, 
                                        encoder_output_size
        '''
        sequence_features, (hn, cn) = self.lstm(visual_features)

        # This is the combination result of visual features and sequence features
        visual_sequence_features = torch.cat((visual_features.view(sequence_size[0], sequence_size[1], -1), sequence_features), 2)
        visual_sequence_features = self.linear_visual_sequence(visual_sequence_features)
        visual_sequence_features = self.dropout(visual_sequence_features)
        visual_sequence_features = self.bn_visual_sequence(visual_sequence_features.contiguous().view(-1, self.hidden_size*(2 if self.is_bidirectional else 1)))
        visual_sequence_features = visual_sequence_features.view(sequence_size[0], sequence_size[1], -1)

        return visual_sequence_features, (hn, cn)

    def get_parameters(self):
        return list(self.parameters())
    
    def init_weight(self):
        self.linear.weight.data.normal_(0.0, 0.02)
        self.linear.bias.data.fill_(0)

class Decoder(nn.Module):
    def __init__(self, embedding_size, hidden_size, num_layers, vocabulary, is_bidirectional, device):
        super(Decoder, self).__init__()
        
        self.embedding_size = embedding_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers 
        self.vocabulary = vocabulary 
        self.is_bidirectional = is_bidirectional
        self.device = device

        # Layer Construction ###################################################
        # https://pytorch.org/docs/stable/nn.html#linear
        '''
            the 'linear_encoding_to_embedding' layer change the dimension 
            size of the embedding representation from the encoding

            encoding ---> embedding
        '''
        self.linear_encoding_to_embedding = nn.Linear(in_features = hidden_size*(2 if is_bidirectional else 1), 
                                out_features = hidden_size)
        
        self.embedding = nn.Embedding(num_embeddings = len(vocabulary), 
                                      embedding_dim = embedding_size)
        
        self.dropout_visual_story_feature = nn.Dropout(p=0.5)
        
        self.dropout_sequence = nn.Dropout(p=0.5)

        self.dropout_embedding = nn.Dropout(p=0.1)
        
        '''
            Arguments:
                - embedding_size: is the output size of the transformation from text to embedding
                                  (input) word index ---> (process) embedding_layer ---> (output) embedding_size
        '''
        self.lstm = nn.LSTM(input_size = embedding_size+hidden_size, 
                            hidden_size = hidden_size, 
                            num_layers = num_layers, 
                            batch_first=True, 
                            dropout=0.5)
        
        '''
            The 'linear_lstm_to_vocabulary' layer change the dimension size
            from lstm output to the length of vocabulary (number of word in vocabulary)

            lstm output ---> vocabulary size
        ''' 
        self.linear_lstm_to_vocabulary = nn.Linear(in_features = hidden_size,
                                                   out_features = len(vocabulary)) 
        
        self.softmax = nn.Softmax(0)

        # END OF Layer Construction ############################################

        # Weight initialization ################################################
        self.start_vec = torch.zeros([1, len(vocabulary)], dtype=torch.float32)
        self.start_vec[0][1] = 10000
        self.start_vec = self.start_vec.to(self.device)
        # END OF Weight initialization #########################################

        self.init_weights()

    def init_weights(self):
        self.linear_encoding_to_embedding.weight.data.normal_(0.0, 0.02)
        self.linear_encoding_to_embedding.bias.data.fill_(0)
        self.linear_lstm_to_vocabulary.weight.data.normal_(0.0, 0.02)
        self.linear_lstm_to_vocabulary.bias.data.fill_(0) 

    def get_parameters(self):
        return list(self.parameters())
    
    def init_hidden(self):
        h0 = torch.zeros(1 * self.num_layers, 1, self.hidden_size).to(self.device)
        c0 = torch.zeros(1 * self.num_layers, 1, self.hidden_size).to(self.device)    
        return (h0, c0)

    def forward(self, visual_sequence_features, text_caption, caption_length):
        # Visual Representation ################################################
        visual_sequence_features = self.linear_encoding_to_embedding(input = visual_sequence_features)
        visual_sequence_features = self.dropout_visual_story_feature(visual_sequence_features)
        visual_sequence_features = nn.functional.relu(visual_sequence_features) # Dimension: [5, hidden size]
        # Dimension: [5, max length text, hidden size], this resize purpose is 
        # to make first 2 dimension of visual and textual are same
        visual_sequence_features = visual_sequence_features.unsqueeze(1).expand(-1, np.amax(caption_length), -1) 
        
        # Textual Representation ###############################################
        text_embedding = self.embedding(text_caption)
        text_embedding = self.dropout_embedding(text_embedding)

        # Features Combining ###################################################
        features_combining = torch.cat((visual_sequence_features, text_embedding), 2)

        outputs = []
        (hn, cn) = self.init_hidden() 

        for i, length in enumerate(caption_length):
            lstm_input = features_combining[i][0:length - 1]
            lstm_input = lstm_input.unsqueeze(0)
            output, (hn, cn) = self.lstm(lstm_input, (hn, cn))
            output = self.dropout_sequence(output)
            output = self.linear_lstm_to_vocabulary(output[0])
            output = torch.cat((self.start_vec, output), 0)
            outputs.append(output)
        
        return outputs       

