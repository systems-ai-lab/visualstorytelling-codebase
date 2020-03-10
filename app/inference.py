import argparse
import yaml
import torch
import torch.nn as nn
from torchvision import transforms
from PIL import Image
from dataset import VIST
from models.CustomModel import Encoder, Decoder
import os, fnmatch
import numpy as np
import time
import json
import subprocess
import ntpath

def main(args):
    ## DO NOT CHANGE! this code
    # configuration YAML formatted file, default configuration is ./config/
    with open(args.config, 'r') as ymlfile:
        cfg = yaml.load(ymlfile) # please use cfg['config_name'] to use
    
    # device configuration
    device = torch.device('cuda:'+str(cfg['inference']['gpu_id']) \
        if torch.cuda.is_available() else 'cpu')
    
    # define the directory
    current_time = time.strftime("%Y%m%d_%H%M%S")
    
    inference_id = ntpath.basename(args.config)
    inference_id = os.path.splitext(inference_id)[0]
    
    result_path = os.path.join(cfg['logging']['result_dir'], inference_id, cfg['model']['id_experiment'], cfg['model']['id_model'])
    if not os.path.exists(result_path):
        os.makedirs(result_path)
    
    # transformation definition
    test_transform =    transforms.Compose([
                        transforms.Resize(cfg['dataset']['image_resize'], 
                        interpolation=Image.LANCZOS),
                        transforms.ToTensor(),
                        transforms.Normalize((0.485, 0.456, 0.406),
                                            (0.229, 0.224, 0.225))])
    
    # loading test dataset
    test_dataset = VIST(dataset_dir=cfg['dataset']['directory'],\
        vocabulary_treshold=cfg['dataset']['vocabulary_treshold'],\
            type='test',\
                transform= test_transform)
    
    # dataset test loader
    data_test_loader = torch.utils.data.DataLoader(dataset=test_dataset,\
        batch_size=cfg['inference']['batch_size'],\
            shuffle=cfg['dataset']['shuffle'],\
                num_workers=cfg['dataset']['num_workers'],\
                    collate_fn=test_dataset.collate_fn)
    
    # model object instantiation
    encoder = Encoder(encoder_output_size = cfg['model']['encoder']['linear_visual_features']['out_size'],\
                hidden_size = cfg['model']['encoder']['lstm']['hidden_size'],\
                    num_layers = cfg['model']['encoder']['lstm']['num_layers'],\
                        is_bidirectional = cfg['model']['encoder']['lstm']['bidirectional']).to(device)
    
    decoder = Decoder(embedding_size = cfg['model']['decoder']['lstm']['embeed_size'], \
                        hidden_size = cfg['model']['decoder']['lstm']['hidden_size'],\
                            num_layers = cfg['model']['decoder']['lstm']['num_layers'], \
                                vocabulary = test_dataset.get_vocabulary(),\
                                    is_bidirectional = cfg['model']['encoder']['lstm']['bidirectional'], \
                                        device = device).to(device)

    # loading pre-trained model
    encoder_file = '*-encoder-'+cfg['model']['id_model']+'.pkl'
    decoder_file = '*-decoder-'+cfg['model']['id_model']+'.pkl'
    listOfFiles = os.listdir(os.path.join(cfg['model']['dir'], cfg['model']['id_experiment']))
    
    # encoder path file
    for entry in listOfFiles:
        if fnmatch.fnmatch(entry, encoder_file):
            encoder_path = os.path.join(cfg['model']['dir'], cfg['model']['id_experiment'], entry)
    
    # decoder path file
    for entry in listOfFiles:
        if fnmatch.fnmatch(entry, decoder_file):
            decoder_path = os.path.join(cfg['model']['dir'], cfg['model']['id_experiment'], entry)
    
    # load state model from pre-trained model
    encoder.load_state_dict(torch.load(encoder_path, map_location=torch.device(device)))
    decoder.load_state_dict(torch.load(decoder_path, map_location=torch.device(device)))
    encoder.eval().to(device)
    decoder.eval().to(device)

    criterion = nn.CrossEntropyLoss()

    results = []
    total_step = len(data_test_loader)
    avg_loss = 0.0
    for bi, (image_stories, targets_set, lengths_set, photo_sequence_set, album_ids_set) in enumerate(data_test_loader):
        
        loss = 0
        images = torch.stack(image_stories).to(device)
        features, _ = encoder(images)

        for si, (feature, captions, lengths, photo_sequence, album_ids) in enumerate(zip(features, targets_set, lengths_set, photo_sequence_set, album_ids_set)):
            captions = captions.to(device)
            outputs = decoder(feature, captions, lengths)

            for sj, result in enumerate(zip(outputs, captions, lengths)):
                loss += criterion(result[0], result[1][0:result[2]])
            
            inference_results = decoder.inference(feature) # feature: [5, 2048]

            sentences = []
            target_sentences = []

            for i, result in enumerate(inference_results):
                words = []
                for word_id in result:
                    word = test_dataset.get_vocabulary().idx2word[word_id.item()]
                    words.append(word)
                    if word == '<end>':
                        break

                try:
                    words.remove('<start>')
                except Exception:
                    pass

                try:
                    words.remove('<end>')
                except Exception:
                    pass

                sentences.append(' '.join(words))
            
            result = {}
            result["duplicated"] = False
            result["album_id"] = album_ids[0]
            result["photo_sequence"] = photo_sequence
            result["story_text_normalized"] = sentences[0] + " " + sentences[1] + " " + sentences[2] + " " + sentences[3] + " " + sentences[4]

            results.append(result)

        avg_loss += loss.item()
        loss /= (cfg['inference']['batch_size'] * 5)

        # Print log info
        if bi % cfg['logging']['print_interval'] == 0:
            print('Step [%d/%d], Loss: %.4f, Perplexity: %5.4f' %(bi, total_step, loss.item(), np.exp(loss.item())))

    avg_loss /= (cfg['inference']['batch_size'] * total_step * 5)
    print('Average Loss: %.4f, Average Perplexity: %5.4f' %(avg_loss, np.exp(avg_loss)))

    for i in reversed(range(len(results))):
        if not results[i]["duplicated"]:
            for j in range(i):
                if np.array_equal(results[i]["photo_sequence"], results[j]["photo_sequence"]):
                    results[j]["duplicated"] = True
    
    filtered_res = []
    for result in results:
        if not result["duplicated"]:
            del result["duplicated"]
            filtered_res.append(result)
    
    print("Total story size : %d" %(len(filtered_res)))

    # Evaluation Area ----------------------------------------------------------
    evaluation_info = {}
    evaluation_info["version"] = "initial version"

    output = {}
    output["team_name"] = "SnuBiVtt"
    output["evaluation_info"] = evaluation_info
    output["output_stories"] = filtered_res

    with open(result_path+"/"+current_time+".json", "w") as json_file:
        json_file.write(json.dumps(output))

    subprocess.call(["java", "-jar", cfg['inference']['evaluation_dir'] + "runnable_jar/EvalVIST.jar", "-testFile", result_path+"/"+current_time+".json", "-gsFile", test_dataset.get_sis_file('test')])
    

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, help='specify the file configuration, YAML formatted')

    args = parser.parse_args()
    main(args)

