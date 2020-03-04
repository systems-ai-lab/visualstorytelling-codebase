import argparse
import torch
import torch.nn as nn
from dataset import VIST
from torchvision import transforms
from PIL import Image
from models.CustomModel import Encoder, Decoder
import yaml
from helper import Logger
import wandb

def main(args):

    ## DO NOT CHANGE! this code
    # configuration YAML formatted file, default configuration is ./config/
    with open(args.config, 'r') as ymlfile:
        cfg = yaml.load(ymlfile) # please use cfg['config_name'] to use
    
    # device configuration
    device = torch.device('cuda:'+str(cfg['training']['gpu_id']) \
        if torch.cuda.is_available() else 'cpu')

    # logger configuration
    logger = Logger(model_dir = cfg['logging']['save_model_dir'], \
        tensorboard_dir = cfg['logging']['tensorboard'], \
            file_config = args.config)
    
    if cfg['logging']['wandb']:
        wandb.init(project=cfg['logging']['wandb-project'])

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

    # save the model structure on tensorboard
    # images,_,_,_,_ = next(iter(data_train_loader))
    # input_encoder = torch.stack(images).to(device)
    # logger.tensorboard_graph(model=encoder, input_to_model= input_encoder)

    # loss function definition
    criterion = nn.CrossEntropyLoss()

    # optimization definition
    optimizer = torch.optim.Adam(parameter, lr=cfg['training']['learning_rate'],\
        weight_decay=cfg['training']['weight_decay'])

    # END OF Training Configuration ############################################

    # logging the model in wandb
    if cfg['logging']['wandb']:
        wandb.watch(encoder)
        wandb.watch(decoder)
    
    min_avg_loss = float("inf")
    overfit_warn = 0
    all_step_train = 0
    all_step_val = 0

    for epoch in range(cfg['training']['epoch']): # epoch itteration ###########
        
        # training phase #######################################################
        encoder.train()
        decoder.train()
        avg_loss_train_per_epoch = 0.0

        for batch_ix, (image_sequence_set, targets_text_set, lengths_set, photo_squence_set, 
            album_ids_set) in enumerate(data_train_loader):
            
            decoder.zero_grad()
            encoder.zero_grad()
            loss_train = 0
            
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
                    loss_train += criterion(zip_output, zip_text_story[0:zip_text_story_length])
            
            avg_loss_train_per_epoch += loss_train.item()
            # the loss value need to devide with the number of batch * number of sequence story (5)
            loss_train = loss_train / (cfg['training']['batch_size'] * 5)
            loss_train.backward()
            optimizer.step()
            
            # add loss value for train process to tensorboard & wandb
            logger.tensorboard_scalar(tag = "loss/train", scalar_value = loss_train.item(), global_step = all_step_train)
            if cfg['logging']['wandb']:
                wandb.log({"Train Loss": loss_train})

            all_step_train +=1

            if batch_ix % cfg['logging']['print_interval'] == 0:
                print('Epoch [%d/%d], Training step [%d/%d], Loss Train: %4f' %(epoch, cfg['training']['epoch'], batch_ix, len(data_train_loader), loss_train.item()))

        # save the trained model every epoch
        logger.save_model(encoder, decoder, epoch)

        # epoch loss
        avg_loss_train_per_epoch = avg_loss_train_per_epoch/(cfg['training']['batch_size'] * 5 * len(data_train_loader))
        print("Average train loss [%d/%d] : %4f" %(epoch, cfg['training']['epoch'], avg_loss_train_per_epoch))

        # validation phase ####################################################
        encoder.eval()
        decoder.eval()
        avg_loss_val_per_epoch = 0.0

        for batch_ix, (image_sequence_set, targets_text_set, lengths_set, photo_squence_set, album_ids_set) in enumerate(data_val_loader):
            
            loss_validation = 0
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
                    loss_validation += criterion(zip_output, zip_text_story[0:zip_text_story_length])
            
            avg_loss_val_per_epoch += loss_validation.item()
            # the loss value need to devide with the number of batch * number of sequence story (5)
            loss_validation = loss_validation / (cfg['training']['batch_size'] * 5)

            # add loss value for train process to tensorboard 
            logger.tensorboard_scalar(tag = "loss/val", scalar_value = loss_validation.item(), global_step = all_step_val)
            if cfg['logging']['wandb']:
                wandb.log({"Validation Loss": loss_validation})

            all_step_val +=1

            if batch_ix % cfg['logging']['print_interval'] == 0:
                print('Epoch [%d/%d], Validation step [%d/%d], Loss Validation: %4f' %(epoch, cfg['training']['epoch'], batch_ix, len(data_val_loader), loss_validation.item()))
        
        avg_loss_val_per_epoch = avg_loss_val_per_epoch/(cfg['training']['batch_size'] * 5 * len(data_val_loader))
        print("Average validation loss [%d/%d] : %4f" %(epoch, cfg['training']['epoch'], avg_loss_val_per_epoch))

        # add loss value to tensorboard & wandb  
        loss_train_val = zip(["train", "validation"], [avg_loss_train_per_epoch, avg_loss_val_per_epoch])
        logger.tensorboard_scalars(tag = "average/loss", scalar_list = loss_train_val,  global_step = epoch)
        if cfg['logging']['wandb']:
            wandb.log({"Average Loss Train": avg_loss_train_per_epoch, "Average Loss Val": avg_loss_val_per_epoch})

        # termination condition
        overfit_warn = overfit_warn + 1 if (min_avg_loss < avg_loss_val_per_epoch) else 0
        min_avg_loss = min(min_avg_loss, avg_loss_val_per_epoch)

        # show the current value of overfitting warning parameter
        logger.tensorboard_scalar(tag = "overfit_warning", scalar_value = overfit_warn, global_step = epoch)

        # the training process will be break if the overfitting warning is greater or equal to the value from config
        if overfit_warn >= cfg['training']['overfit_warning']:
            break

        # END OF epoch itteration ##############################################

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, help='specify the file configuration, YAML formatted')

    args = parser.parse_args()
    main(args)