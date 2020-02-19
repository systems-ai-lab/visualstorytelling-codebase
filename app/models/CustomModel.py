import torch
import torch.nn as nn
import torchvision.models as models

class Encoder(nn.Module):
    '''
        The encoding process flow is by extract the visual features and then
        pass it to the RNN, as the input is sequence of images
    '''
    def __init__(self, encoder_output_size, hidden_size, num_layers):
        super(Encoder, self).__init__()
        
        resnet = models.resnet152(pretrained=True)
        modules = list(resnet.children())[:-1]
        self.resnet = nn.Sequential(*modules)
        for param in self.resnet.parameters():
            param.requires_grad = False

        # https://pytorch.org/docs/stable/nn.html#linear
        self.linear_visual_features = nn.Linear(in_features = resnet.fc.in_features, 
                                out_features = encoder_output_size)
        
        # https://pytorch.org/docs/stable/nn.html#batchnorm1d
        self.bn_visual_features = nn.BatchNorm1d(num_features = encoder_output_size, 
                                                 momentum= 0.01)

        # https://pytorch.org/docs/stable/nn.html#lstm                        
        self.lstm = nn.LSTM(input_size = encoder_output_size, 
                            hidden_size = hidden_size, 
                            num_layers = num_layers, 
                            batch_first=True, 
                            bidirectional=True, 
                            dropout=0.5) 
    
    def forward(self, image_sequence):
        '''
            image_sequence dimension: batch_size, story_length, channels,
                                      width, height
        '''
        sequence_size = image_sequence.size()
        image_sequence = image_sequence.view(-1, sequence_size[2], 
                                                 sequence_size[3], 
                                                 sequence_size[4])
        '''
            image_sequence dimension:   batch_size * story_length, 
                                        channel, width, height
        '''
        visual_features = self.resnet(image_sequence)
        visual_features = self.linear_visual_features(visual_features.squeeze())
        visual_features = self.bn_visual_features(visual_features)
        
        visual_features = visual_features.view(sequence_size[0], sequence_size[1], -1)
        '''
            returning back the batch and story length dimension
            visual_features dimension:  batch_size, story_length, 
                                        encoder_output_size
        '''
        sequence_features, (hn, cn) = self.lstm(visual_features)

        # This is the combination result of visual features and sequence features
        visual_sequence_features = torch.cat((visual_features.view(sequence_size[0], sequence_size[1], -1), sequence_features), 2)

        print(visual_sequence_features.shape)
        exit()



        return

    def get_parameters(self):
        return list(self.parameters())
    
    def init_weight(self):
        self.linear.weight.data.normal_(0.0, 0.02)
        self.linear.bias.data.fill_(0)

class Decoder(nn.Module):
    def __init__(self):
        super(Decoder, self)
    
    def get_parameters(self):
        return list(self.parameters())

    def forward(self, data):
        pass
