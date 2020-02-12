# TODO:
# check the data is ready or not including directory, images, text-annotation file, and vocabulary file
# 
import os
import os.path
import json
from vocabulary import GenerateVocabulary


class VIST():
    def __init__(self, dataset_dir, vocabulary_treshold=10):
        self.dataset_dir = dataset_dir # str: main dataset directory
        self.vocabulary_treshold = vocabulary_treshold # int: get value minimum treshold of vocabulary from SIS dataset 
        self._check_exists()
    
    def _check_exists(self):
        # check exist main dataset directory
        if not os.path.exists(self.dataset_dir):
            raise RuntimeError('Main dataset directory is not found: ', self.dataset_dir) 
       
        # check exist dataset structure directory
        if not os.path.exists(os.path.join(self.dataset_dir, 'images')):
            raise RuntimeError('"images" directory is not found')
       
        if not os.path.exists(os.path.join(self.dataset_dir, 'text-annotation')):
            raise RuntimeError('"text-annotation" directory is not found')
       
        if not os.path.exists(os.path.join(self.dataset_dir,'vocabulary', 'vocabulary-'+str(self.vocabulary_treshold)+'.pkl')):
            self.generate_vocabulary()
    
    def generate_vocabulary(self):
        if not os.path.exists(os.path.join(self.dataset_dir,'vocabulary')):
            os.mkdir(os.path.join(self.dataset_dir, 'vocabulary')) 

        # generating vocabulary
        output_vocabulary = os.path.join(self.dataset_dir,'vocabulary', 'vocabulary-'+str(self.vocabulary_treshold)+'.pkl')
        generate = GenerateVocabulary(self.sis_formatting(), self.vocabulary_treshold, output_vocabulary)


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
