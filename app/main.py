import argparse
import torch
import torch.nn as nn
from dataset import VIST
from torchvision import transforms
from PIL import Image
from models.CustomModel import Encoder, Decoder
import yaml

def main(args):

    ## DO NOT CHANGE! this code
    # configuration YAML formatted file, default configuration is ./config/
    with open(args.config, 'r') as ymlfile:
        cfg = yaml.load(ymlfile) # please use cfg['config_name'] to use
    
    # device configuration
    device = torch.device('cuda:'+str(cfg['training']['gpu_id']) \
        if torch.cuda.is_available() else 'cpu')

    ## END OF DO NOT CHANGE! this code
    
    # Transformation definition
    val_transform = transforms.Compose([
            transforms.Resize(cfg['dataset']['image_resize'],
                                interpolation=Image.LANCZOS),
            transforms.ToTensor(),
            transforms.Normalize((0.485, 0.456, 0.406),
                                (0.229, 0.224, 0.225))])

    
    # Dataset Configuration ####################################################
    val_dataset = VIST(dataset_dir=cfg['dataset']['directory'], 
                    vocabulary_treshold=cfg['dataset']['vocabulary_treshold'], 
                    type='val', 
                    transform= val_transform)

    data_val_loader = torch.utils.data.DataLoader(dataset=val_dataset,
                                    batch_size=cfg['training']['batch_size'],
                                    shuffle=True, num_workers=0,
                                    collate_fn=val_dataset.collate_fn)
    # END OF Dataset Configuration #############################################
    
    # Training Configuration ###################################################
    # model object instantiation
    encoder = Encoder(encoder_output_size = cfg['model']['encoder']['linear_visual_features']['out_size'],\
        hidden_size = cfg['model']['encoder']['lstm']['hidden_size'],\
        num_layers = cfg['model']['encoder']['lstm']['num_layers']).to(device)
    parameter = encoder.get_parameters()

    # loss function definition
    criterion = nn.CrossEntropyLoss()

    # optimization definition
    optimizer = torch.optim.Adam(parameter, lr=cfg['training']['learning_rate'], 
                                weight_decay=cfg['training']['weight_decay'])

    # END OF Training Configuration ############################################

    for epoch in range(cfg['training']['epoch']): # epoch itteration ###########
        # print(epoch)
        for bi, (image_sequence, targets_set, lengths_set, photo_squence_set, 
            album_ids_set) in enumerate(data_val_loader):
            
            image_sequence = torch.stack(image_sequence).to(device) 
            '''
            image_stories dimension:
                batch_size, story, channels, width, height
            '''
            features = encoder(image_sequence)
        
        #         print(photo_squence_set)
        # END OF epoch itteration ##############################################
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, help='specify the file configuration, YAML formatted')

    args = parser.parse_args()
    main(args)