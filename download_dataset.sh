#!/bin/bash

# AUTHOR: Rizal Setya Perdana (rizalespe@ub.ac.id) DATE: 2020-02-11

# About this script: the purpose of this script is to  
# automatically downloading the Visual Storytelling (VIST) Dataset 
# from http://visionandlanguage.net/VIST/dataset.html. Please cite 
# https://arxiv.org/abs/1604.03968 if you use this script.

# NOTES: we have to change the download images link location and reupload 
# to our Google Drive due to the permision issues from the GDOWN package.
# Our dataset version is already resized to 256x256.   

# ARGUMENT: to specify the download location, add -d <DIR> argument for the dataset location.

# The structure of the directory is as follow:
# 
# dataset
# ├── images
# │   ├── test
# │   │   ├──images_1.jpg
# │   │   ├──images_n.jpg
# │   ├── train
# │   │   ├──images_1.jpg
# │   │   ├──images_n.jpg
# │   └── val
# │       ├──images_1.jpg
# │       ├──images_n.jpg
# └── text-annotation
#     ├── dii
#     │   ├── test.description-in-isolation.json
#     │   ├── train.description-in-isolation.json
#     │   └── val.description-in-isolation.json
#     └── sis
#         ├── test.story-in-sequence.json
#         ├── train.story-in-sequence.json
#         └── val.story-in-sequence.json

# TODO:
# 1. Determining the base path of the dataset location (Finish)
#     - create directory for save the original file of dataset (Finish)
#     - create directory for text-annotation and images-annotation (Finish)
# 2. Extracting the zip file
#     - placing the dataset in the created directory (Finish)
#     - remove the directory of contained original compressed file 
# 3. Resizing all of the images file (pending, we already change the 
#    download images link to resized version)

. ~/.bashrc # make sure all of the variable of the user can same as with user login

# Checking the dependency packages
if python -c 'import gdown;'; then
    echo -e '[INFO] Checking the package dependency... OK'
else
    echo -e '[ERROR] You need to install gdown package by typing: "pip install gdown"'
    exit 1
fi

# Recieve user specify argument
while [ "$1" != "" ]; do
    case $1 in
        -d | --directory ) shift 
                                specify_download_dir=$1 ;;
    esac
    shift
done

echo "[INFO] Preparing the directory sturcture..."

# function for preparing directory structure
preparing_the_directory () {
    cd dataset
    mkdir -p "text-annotation" # location of text annotation files after extracted
    mkdir -p "images" # location of images files after extracted

    mkdir -p "original" # location of original downloaded files, this directory will be removed after all finished downloaded
    cd original
    mkdir -p "text-annotation"
    mkdir -p "images"
}

if test -z "$specify_download_dir" # check if user specify the location of dataset directory 
then
    # the condition when user DO NOT specify the download location
    cd $(pwd)
    cd ..

    if [[ -d "$(pwd)/dataset" ]] # check if the "dataset" directory is exist
    then
        # the condition which the directory is exist
        echo -e "[INFO] '$(pwd)/dataset' is exists in your system, and the dataset will downloaded here."
        
        preparing_the_directory

    else
        mkdir -p "dataset"
        # the condition which the directory is not exist
        download_dir=$(pwd)"/dataset"
        echo -e "[INFO] The dataset will downloaded in: '$download_dir'"

        preparing_the_directory
    fi
        
else
    # the condition when user SPECIFY the download location
    echo -e "[INFO] The dataset will downloaded in: $specify_download_dir""dataset"
    mkdir -p "$specify_download_dir""dataset"
    cd $specify_download_dir
    preparing_the_directory
fi

echo "[INFO] The directory preparation is finish..."
cd text-annotation

download_text_annotation () {

    wget -O dii.tar.gz http://visionandlanguage.net/VIST/json_files/description-in-isolation/DII-with-labels.tar.gz
    wget -O sis.tar.gz http://visionandlanguage.net/VIST/json_files/story-in-sequence/SIS-with-labels.tar.gz

    local file_dii='dii.tar.gz'
    local file_sis='sis.tar.gz'

    # check the files succesfully downloaded
    if test -f $file_dii; then
        echo -e "[INFO] File $file_dii is downloaded succesfully... OK"
    fi

    if test -f $file_sis; then
        echo -e "[INFO] File $file_sis is downloaded succesfully... OK"
    fi

    # extracting the text-annotation file
    tar -xf $file_dii -C ../../text-annotation/
    tar -xf $file_sis -C ../../text-annotation/

    echo -e "[INFO] Extracting text-annotation files succesfully... OK"
}

download_images_annotation(){
    cd ../images/
    
    echo -e "[INFO] Starting to download the TEST DATASET (resized version) ..."
    gdown https://drive.google.com/uc?id=1zv-9NHLLzWCANumdsNtwDK1WV549b270 # test dataset
    echo -e "[INFO] Starting to download the TRAIN DATASET (resized version) ..."
    gdown https://drive.google.com/uc?id=1KO-MzfAW62jI-yL_hEHsxjA1DbvvjAaT # train dataset
    echo -e "[INFO] Starting to download the VALIDATION DATASET  (resized version) ..."
    gdown https://drive.google.com/uc?id=1wAL_Th_skQlLyQIxo-xExJF7Zf0sa_1- # val dataset

    unzip test.zip -d ../../images/ # extract test images to directory images
    unzip train.zip -d ../../images/ # extract train images to directory images
    unzip val.zip -d ../../images/ # extract validation images to directory images
    rm -rf ../../images/__MACOSX/ # remove unnecessary directory
    rm -rf ../../original/ # remove the original directory containing zip file
}

echo -e "[INFO] Starting to download text-annotation files..."
download_text_annotation
echo -e "[INFO] Starting to download images files..."
download_images_annotation
echo -e "[INFO] Download the datases is finish"
