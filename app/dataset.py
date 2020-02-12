# TODO:
# check the data is ready or not including directory, images, text-annotation file, and vocabulary file
# 
import os
import os.path
import json
import torch
import torch.utils.data as data
from PIL import Image
import pickle

from vocabulary import GenerateVocabulary

class VIST(data.Dataset):

    def __init__(self, dataset_dir, vocabulary_treshold=10, type='train', transform=None):
        self.dataset_dir = dataset_dir # str: main dataset directory
        self.vocabulary_treshold = vocabulary_treshold # int: get value minimum treshold of vocabulary from SIS dataset 
        self._check_exists()
        self.vist = self.sis_formatting(types=type)
        self.ids = list(self.vist['stories'].keys())
        self.transform = transform
        self.type = type

    def _check_exists(self):
        # check exist main dataset directory
        if not os.path.exists(self.dataset_dir):
            raise RuntimeError('Main dataset directory is not found: ', self.dataset_dir) 
       
        # check exist dataset structure directory
        self.image_dir = os.path.join(self.dataset_dir, 'images')
        if not os.path.exists(self.image_dir):
            raise RuntimeError('"images" directory is not found')
        self.annotation_dir = os.path.join(self.dataset_dir, 'text-annotation')
        if not os.path.exists(self.annotation_dir):
            raise RuntimeError('"text-annotation" directory is not found')
        self.vocabulary_file = os.path.join(self.dataset_dir,'vocabulary', 'vocabulary-'+str(self.vocabulary_treshold)+'.pkl')
        if not os.path.exists(self.vocabulary_file):
            self.generate_vocabulary()
        else:
            with open(self.vocabulary_file, 'rb') as f:
                self.vocab = pickle.load(f) 
    
    def generate_vocabulary(self):
        if not os.path.exists(os.path.join(self.dataset_dir,'vocabulary')):
            os.mkdir(os.path.join(self.dataset_dir, 'vocabulary')) 

        # generating vocabulary
        generate = GenerateVocabulary(self.sis_formatting(), self.vocabulary_treshold, self.vocabulary_file)
        with open(self.vocabulary_file, 'rb') as f:
            self.vocab = pickle.load(f)

    def sis_formatting(self, types='train'):
        # This function intended to formatting the text-annotation JSON file of VIST dataset

        sis_file_type ={'train':'train.story-in-sequence.json', 
                        'val':'val.story-in-sequence.json', 
                        'test':'test.story-in-sequence.json'}

        sis_file = os.path.join(self.dataset_dir, 'text-annotation', 'sis', sis_file_type[types])
        
        if not os.path.exists(sis_file):
            raise RuntimeError('"File annotation is not found')
        
        sis_dataset = json.load(open(sis_file, 'r'))

        images = {}
        stories = {}

        if 'images' in sis_dataset:
            for image in sis_dataset['images']:
                images[image['id']] = image

        if 'annotations' in sis_dataset:
            annotations = sis_dataset['annotations']
            for annotation in annotations:
                story_id = annotation[0]['story_id']
                stories[story_id] = stories.get(story_id, []) + [annotation[0]]

        data = {'images': images, 'stories': stories}
        return data

    def __getitem__(self, index):
        vist = self.vist
        vocab = self.vocab
        story_id = self.ids[index]

        targets = []
        images = []
        photo_sequence = []
        album_ids = []

        story = vist['stories'][story_id]
        image_formats = ['.jpg', '.gif', '.png', '.bmp']

        for annotation in story:
            storylet_id = annotation["storylet_id"]
            image = Image.new('RGB', (256, 256))
            image_id = annotation["photo_flickr_id"]
            photo_sequence.append(image_id)
            album_ids.append(annotation["album_id"])
            
            for image_format in image_formats:
                try:
                    image = Image.open(os.path.join(self.image_dir, self.type, str(image_id) + image_format)).convert('RGB')
                except Exception:
                    continue
    
            if self.transform is not None:
                image = self.transform(image)
            

            images.append(image)
           
            text = annotation["text"]
            tokens = []
            try:
                tokens = nltk.tokenize.word_tokenize(text.lower())
            except Exception:
                pass

            caption = []
            caption.append(vocab('<start>'))
            caption.extend([vocab(token) for token in tokens])
            caption.append(vocab('<end>'))
            target = torch.Tensor(caption)
            targets.append(target)

        return torch.stack(images), targets, photo_sequence, album_ids

    def __len__(self):
        return len(self.ids)

    def collate_fn(self, data):
        image_stories, caption_stories, photo_sequence_set, album_ids_set = zip(*data)

        targets_set = []
        lengths_set = []

        for captions in caption_stories:
            lengths = [len(cap) for cap in captions]
            targets = torch.zeros(len(captions), max(lengths)).long()
            for i, cap in enumerate(captions):
                end = lengths[i]
                targets[i, :end] = cap[:end]

            targets_set.append(targets)
            lengths_set.append(lengths)

        return image_stories, targets_set, lengths_set, photo_sequence_set, album_ids_set