### Update
2020-11-18: add [Jupyter notebook](https://github.com/systems-ai-lab/visualstorytelling-codebase/blob/master/display.ipynb) for displaying VIST dataset

# Visual Storytelling Codebase
This repository containing source code and documentation in the process of the visual storytelling task development. Visual storytelling is a task transforming a sequence of images into a coherent story. Different from another image-to-text task such as image captioning, the visual storytelling task has a special challenge due to the nature of the story generated not only describe the literal object in image representation but also non-visual concept should appear in text generated story. 

This codebase including both source code written in Python and also documentation for each step. Included in this repository:
1. [Downloading the dataset](https://github.com/systems-ai-lab/visualstorytelling-codebase/blob/master/documentation/downloading-the-dataset.md)
2. [Generating the text vocabulary](https://github.com/systems-ai-lab/visualstorytelling-codebase/blob/master/documentation/generating-text-vocabulary.md)

To run the training process, execute the ```main.py``` file following by the configuration file as follow:
```
$ cd app
$ python main.py --config config/sample-config.yml
```

To run the inference process (generating stories from trained model) execute ```inference.py``` as follow:
```
$ cd app
$ python inference.py --config config/sample-config-inference.yml
```

To run the evaluation, (generating score from generated stories) execute ```evaluation.py``` as follow:
```
$ cd app
$ python evaluation.py --config config/sample-config-evaluation.yml
```

For evaluation, we need the following code:
1. VIST API: https://github.com/lichengunc/vist_api
2. VIST Eval: https://github.com/lichengunc/vist_eval.git
3. VIST Challenge NAACL 2018: https://github.com/windx0303/VIST-Challenge-NAACL-2018
