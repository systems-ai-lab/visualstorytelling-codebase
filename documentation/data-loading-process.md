# Data Loading Process

If you use the dataset downloader script [download_dataset.sh](https://github.com/systems-ai-lab/visualstorytelling-codebase/blob/master/app/download_dataset.sh), this script will arrange the dataset directory as follow:

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

If the dataset is downloaded with the required structure, we can instantiate the [class VIST](https://github.com/systems-ai-lab/visualstorytelling-codebase/blob/c76c394a713117675dab1ebc56ab14856b40781b/app/dataset.py#L17) located on the /app/dataset.py file. VIST class is a user definied class and a subclass from [torch.utils.data.Dataset](https://pytorch.org/docs/stable/data.html#torch.utils.data.Dataset) with the following arguments:

