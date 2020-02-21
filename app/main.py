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
    train_transform = transforms.Compose([
        transforms.RandomCrop(cfg['dataset']['image_resize']),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.485, 0.456, 0.406),
                             (0.229, 0.224, 0.225))])

    val_transform = transforms.Compose([
            transforms.Resize(cfg['dataset']['image_resize'],
                                interpolation=Image.LANCZOS),
            transforms.ToTensor(),
            transforms.Normalize((0.485, 0.456, 0.406),
                                (0.229, 0.224, 0.225))])

    
    # Dataset Configuration ####################################################
    train_dataset = VIST(dataset_dir=cfg['dataset']['directory'],\
        vocabulary_treshold=cfg['dataset']['vocabulary_treshold'],\
            type='train',\
                transform= train_transform)

    val_dataset = VIST(dataset_dir=cfg['dataset']['directory'],\
        vocabulary_treshold=cfg['dataset']['vocabulary_treshold'],\
            type='val',\
                transform= val_transform)

    data_train_loader = torch.utils.data.DataLoader(dataset=train_dataset,\
        batch_size=cfg['training']['batch_size'],\
            shuffle=cfg['dataset']['shuffle'],\
                num_workers=cfg['dataset']['num_workers'],\
                    collate_fn=train_dataset.collate_fn)

    data_val_loader = torch.utils.data.DataLoader(dataset=val_dataset, \
        batch_size=cfg['training']['batch_size'],\
            shuffle=cfg['dataset']['shuffle'],\
                num_workers=cfg['dataset']['num_workers'],\
                    collate_fn=val_dataset.collate_fn)
    # END OF Dataset Configuration #############################################
    
    # Training Configuration ###################################################
    # model object instantiation
    encoder = Encoder(encoder_output_size = cfg['model']['encoder']['linear_visual_features']['out_size'],\
                hidden_size = cfg['model']['encoder']['lstm']['hidden_size'],\
                    num_layers = cfg['model']['encoder']['lstm']['num_layers'],\
                        is_bidirectional = cfg['model']['encoder']['lstm']['bidirectional']).to(device)
    
    decoder = Decoder(embedding_size = cfg['model']['decoder']['lstm']['embeed_size'], \
                        hidden_size = cfg['model']['decoder']['lstm']['hidden_size'],\
                            num_layers = cfg['model']['decoder']['lstm']['num_layers'], \
                                vocabulary = train_dataset.get_vocabulary(),\
                                    is_bidirectional = cfg['model']['encoder']['lstm']['bidirectional'], \
                                        device = device).to(device)
    
    parameter = encoder.get_parameters() + decoder.get_parameters()

    # loss function definition
    criterion = nn.CrossEntropyLoss()

    # optimization definition
    optimizer = torch.optim.Adam(parameter, lr=cfg['training']['learning_rate'],\
        weight_decay=cfg['training']['weight_decay'])

    # END OF Training Configuration ############################################

    for epoch in range(cfg['training']['epoch']): # epoch itteration ###########
        
        encoder.train()
        decoder.train()
        avg_loss = 0.0

        for batch_ix, (image_sequence_set, targets_text_set, lengths_set, photo_squence_set, 
            album_ids_set) in enumerate(data_train_loader):
            
            decoder.zero_grad()
            encoder.zero_grad()
            loss = 0
            
            image_sequence_set = torch.stack(image_sequence_set).to(device) 
            '''
            image_sequence dimension:
                batch_size, story_sequence_length, image_channels, width, height
            '''
            # encoding sequence of images (visual representation)
            visual_sequence_features, _ = encoder(image_sequence_set)
            
            # pairing image and text data
            visual_textual_pairs = zip(visual_sequence_features, targets_text_set, lengths_set)
    
            # decoding the visual features with the text story
            # this loop process only 1 story for each iteration. It will iterate 'batch_size' times
            for story_ix, (visual_feature, text_story, text_story_length) in enumerate(visual_textual_pairs):
                
                text_story = text_story.to(device) 
                output = decoder(visual_feature, text_story, text_story_length)

                for sj, (zip_output, zip_text_story, zip_text_story_length) in enumerate(zip(output, text_story, text_story_length)):
                    loss += criterion(zip_output, zip_text_story[0:zip_text_story_length])
            
            avg_loss += loss.item()
            # the loss value need to devide with the number of batch * number of sequence story (5)
            loss = loss / (cfg['training']['batch_size'] * 5)
            loss.backward()
            optimizer.step()

            if batch_ix % cfg['logging']['interval'] == 0:
                print('Epoch [%d/%d], Training step [%d/%d], Loss Train: %4f' %(epoch, cfg['training']['epoch'], batch_ix, len(data_train_loader), loss.item()))
        
        #         print(photo_squence_set)
        # END OF epoch itteration ##############################################
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, help='specify the file configuration, YAML formatted')

    args = parser.parse_args()
    main(args)