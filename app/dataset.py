# AUTHOR: Rizal Setya Perdana (rizalespe@ub.ac.id)

# About this script: This Python script is mainly used as custom dataset wrapper 
# and subclass of torch.utils.data.Dataset.


import os
import os.path
import json
import torch
import torch.utils.data as data
from PIL import Image
import pickle
import nltk
from vocabulary import GenerateVocabulary

class VIST(data.Dataset):
    """
        Args:
            - dataset_dir (string): string of path to the dataset directory. 
                In this codebase, the structure of dataset directory is as 
                follow:
                ./dataset
                    ├── images
                    │   ├── test
                    │   │   ├──images_1.jpg
                    │   │   ├──images_n.jpg
                    │   ├── train
                    │   │   ├──images_1.jpg
                    │   │   ├──images_n.jpg
                    │   └── val
                    │       ├──images_1.jpg
                    │       ├──images_n.jpg
                    └── text-annotation
                        ├── dii
                        │   ├── test.description-in-isolation.json
                        │   ├── train.description-in-isolation.json
                        │   └── val.description-in-isolation.json
                        └── sis
                            ├── test.story-in-sequence.json
                            ├── train.story-in-sequence.json
                            └── val.story-in-sequence.json
                You can download and generate the structure and the dataset same
                as above, you can use download_dataset.sh script by executing:
                $ sh download_dataset.sh 
                OR
                $ sh download_dataset.sh -d <DIR>
                if you want to specify the download directory location.
                
            - vocabulary_treshold (int, optional): if you do not specify this
                parameter it set to default value 10. If the vocabulary file is 
                not found in /dataset/vocabulary/ directory, it will generate
                a new vocabulary file. But, if the vocabulary file is already 
                exist, it will use the exsisting file to load the VIST dataset.  
            
            - type (string, optional): this parameter specify what kind of data
                will be loaded that has 3 options [train, test, val]. This
                depend on the need in process of learning

            - transform (object): this parameter is an object from
                torchvision.transforms which has purpose to transform 
                the images data into the tensor

    """
    def __init__(self, dataset_dir, vocabulary_treshold=10, type='train', transform=None):
        self.dataset_dir = dataset_dir
        self.vocabulary_treshold = vocabulary_treshold
        self._check_exists()
        self.vist = self.sis_formatting(types=type)
        self.ids = list(self.vist['stories'].keys())
        self.transform = transform
        self.type = type

    def __getitem__(self, index):
        """Return the data...
            Args: index 
            Returns: 
        """
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

    def _check_exists(self):
        """Checking the requirement, the availability of the dataset, and the 
        directory structure is adequate for the next step or not. If the check
        is fail, it will raise the error runtime.
        """
        # check exist main dataset directory
        if not os.path.exists(self.dataset_dir):
            raise RuntimeError('Main dataset directory is not found: ', 
        self.dataset_dir) 
       
        # check exist dataset structure directory
        self.image_dir = os.path.join(self.dataset_dir, 'images')
        if not os.path.exists(self.image_dir):
            raise RuntimeError('"images" directory is not found')
        
        self.annotation_dir = os.path.join(self.dataset_dir, 'text-annotation')
        if not os.path.exists(self.annotation_dir):
            raise RuntimeError('"text-annotation" directory is not found')
        
        self.vocabulary_file = os.path.join(self.dataset_dir,'vocabulary',
            'vocabulary-'+str(self.vocabulary_treshold)+'.pkl')
        
        if not os.path.exists(self.vocabulary_file):
            self.generate_vocabulary()
        else:
            with open(self.vocabulary_file, 'rb') as f:
                self.vocab = pickle.load(f) 
    
    def generate_vocabulary(self):
        """ If vocabulary file is not exist, this function will be called to
            generate the word vocabulary from SIS text-annotation dataset.
        """
        if not os.path.exists(os.path.join(self.dataset_dir,'vocabulary')):
            os.mkdir(os.path.join(self.dataset_dir, 'vocabulary')) 

        GenerateVocabulary(self.sis_formatting(), 
                    self.vocabulary_treshold, self.vocabulary_file)
        
        with open(self.vocabulary_file, 'rb') as f:
            self.vocab = pickle.load(f)

    def sis_formatting(self, types='train'):
        """This function purpose is reformating from text-annotation JSON file
            to the new data structure. 

            Args: type [train, val, test]. Default: train
            Return: the formatted SIS text-annotation object.

        """
        sis_file_type ={'train':'train.story-in-sequence.json', 
                        'val':'val.story-in-sequence.json', 
                        'test':'test.story-in-sequence.json'}

        sis_file = os.path.join(self.dataset_dir, 'text-annotation', 'sis', 
                    sis_file_type[types])
        
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

    def collate_fn(self, data):
        """This function is to collate the returned data into 5 for each story.
        For each story it contain image_stories, caption, length of caption,
        file image squence, and album code.
        """
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