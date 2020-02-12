# Downloading Visual Storytelling (VIST) Dataset
To download the VIST dataset, we already provide a script downloader [download_dataset.sh](https://github.com/systems-ai-lab/visualstorytelling-codebase/blob/master/script/download_dataset.sh). Download this script and run on terminal/console:
```
$ sh download_dataset.sh -d <DESTINATION_DIRECTORY>
```

The visual storytelling (VIST) dataset is available to download from [VIST dataset](http://visionandlanguage.net/VIST/dataset.html) link. This dataset has two categories include Descriptions of Images-in-Isolation (DII) and Stories of Images-in-Sequence (SIS). The main difference between these two categories is the time context and narrative language. The VIST dataset contain two kind of dataset to download including text annotation and image files.

## Text Annotation Download
To download the text annotation we can download directly from the website link below:
1. [Text Annotation - DII](http://visionandlanguage.net/VIST/json_files/description-in-isolation/DII-with-labels.tar.gz)
2. [Text Annotation - SIS](http://visionandlanguage.net/VIST/json_files/story-in-sequence/SIS-with-labels.tar.gz)

## Image Files Download
There are two ways to download the image files, depending on the environment that we use due to the location of the files is in Google Drive. If we use a GUI based operating system, we can directly access the Google Drive link that available in [VIST homepage](http://visionandlanguage.net/VIST/dataset.html):

### Direct Download Using Browser
#### Train set
1. [train_split.0.tar.gz (22 GB)](https://drive.google.com/open?id=0ByQS_kT8kViSeEpDajIwOUFhaGc)
2. [train_split.1.tar.gz (16 GB)](https://drive.google.com/open?id=0ByQS_kT8kViSZnZPY1dmaHJzMHc)
3. [train_split.2.tar.gz (15 GB)](https://drive.google.com/open?id=0ByQS_kT8kViSb0VjVDJ3am40VVE)
4. [train_split.3.tar.gz (15 GB)](https://drive.google.com/open?id=0ByQS_kT8kViSTmQtd1VfWWFyUHM)
5. [train_split.4.tar.gz (26 GB)](https://drive.google.com/open?id=0ByQS_kT8kViSQ1ozYmlITXlUaDQ)
6. [train_split.5.tar.gz (17 GB)](https://drive.google.com/open?id=0ByQS_kT8kViSTVY1MnFGV0JiVkk)
7. [train_split.6.tar.gz (27 GB)](https://drive.google.com/open?id=0ByQS_kT8kViSYmhmbnp6d2I4a2M)
8. [train_split.7.tar.gz (24 GB)](https://drive.google.com/open?id=0ByQS_kT8kViSZl9aNGVuX0llcEU)
9. [train_split.8.tar.gz (22 GB)](https://drive.google.com/open?id=0ByQS_kT8kViSWXJ3R3hsZllsNVk)
10. [train_split.9.tar.gz (22 GB)](https://drive.google.com/open?id=0ByQS_kT8kViSR2N4cFpweURhTjg)
11. [train_split.10.tar.gz (13 GB)](https://drive.google.com/open?id=0ByQS_kT8kViScllKWnlaVU53Skk)
12. [train_split.11.tar.gz (24 GB)](https://drive.google.com/open?id=0ByQS_kT8kViSV2QxZW1rVXcxT1U)
13. [train_split.12.tar.gz (13 GB)](https://drive.google.com/open?id=0ByQS_kT8kViSNGNPTEFhdGxkMnM)
#### Validation set
[val_images.tar.gz (28 GB)](https://drive.google.com/open?id=0ByQS_kT8kViSWmtRa1lMcG1EaHc)
#### Test set
[test_images.tar.gz (35 GB)](https://drive.google.com/open?id=0ByQS_kT8kViSTHJ0cGxSVW1SRFk)

### Download From CLI
It is not possible to download the files directly using ```wget``` Linux command to Google Drive link. But it is possible by using
[gdown](https://pypi.org/project/gdown/) on Python. Gdown is a special package in Python that allows you to download large files from Google Drive by using link id. First, install gdown using the pip command as follows:
```
pip install gdown
```
After installing the ```gdown``` we can download by the following command from CLI:
#### Train set
```
gdown https://drive.google.com/uc?id=0ByQS_kT8kViSeEpDajIwOUFhaGc
gdown https://drive.google.com/uc?id=0ByQS_kT8kViSZnZPY1dmaHJzMHc
gdown https://drive.google.com/uc?id=0ByQS_kT8kViSb0VjVDJ3am40VVE
gdown https://drive.google.com/uc?id=0ByQS_kT8kViSTmQtd1VfWWFyUHM
gdown https://drive.google.com/uc?id=0ByQS_kT8kViSQ1ozYmlITXlUaDQ
gdown https://drive.google.com/uc?id=0ByQS_kT8kViSTVY1MnFGV0JiVkk
gdown https://drive.google.com/uc?id=0ByQS_kT8kViSYmhmbnp6d2I4a2M
gdown https://drive.google.com/uc?id=0ByQS_kT8kViSZl9aNGVuX0llcEU
gdown https://drive.google.com/uc?id=0ByQS_kT8kViSWXJ3R3hsZllsNVk
gdown https://drive.google.com/uc?id=0ByQS_kT8kViSR2N4cFpweURhTjg
gdown https://drive.google.com/uc?id=0ByQS_kT8kViScllKWnlaVU53Skk
gdown https://drive.google.com/uc?id=0ByQS_kT8kViSV2QxZW1rVXcxT1U
gdown https://drive.google.com/uc?id=0ByQS_kT8kViSNGNPTEFhdGxkMnM
```
#### Validation set
```
gdown https://drive.google.com/uc?id=0ByQS_kT8kViSWmtRa1lMcG1EaHc
```
#### Test set
```
gdown https://drive.google.com/uc?id=0ByQS_kT8kViSTHJ0cGxSVW1SRFk
```
