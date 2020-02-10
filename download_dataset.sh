#!/bin/bash
# Script for downloading VIST (Visual Storytelling Dataset)
: '
    To do list:
    1. Determining the base path of the dataset location (Finish)
        - create directory for save the original file of dataset (Finish)
        - create directory for text-annotation and images-annotation (Finish)
    2. Extracting the zip file
        - placing the dataset in the created directory (Finish)
        - remove the directory of contained original compressed file 
    3. Resizing all of the images file

'
# User specify argument
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
        echo "[INFO] '$(pwd)/dataset' is exists in your system, and the dataset will downloaded here."
        
        preparing_the_directory

    else
        mkdir -p "dataset"
        # the condition which the directory is not exist
        download_dir=$(pwd)"/dataset"
        echo "[INFO] The dataset will downloaded in: '$download_dir'"

        preparing_the_directory
    fi
        
else
    # the condition when user SPECIFY the download location
    echo "[INFO] The dataset will downloaded in: $specify_download_dir""dataset"
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
        echo "[INFO] File $file_dii is downloaded succesfully... OK"
    fi

    if test -f $file_sis; then
        echo "[INFO] File $file_sis is downloaded succesfully... OK"
    fi

    # extracting the text-annotation file
    tar -xf $file_dii -C ../../text-annotation/
    tar -xf $file_sis -C ../../text-annotation/

    echo "[INFO] Extracting text-annotation files succesfully... OK"
    
}

download_images_annotation(){
    cd ../images/

    # Original download link is not working (10/02/2020) :(
    # gdown https://drive.google.com/uc?id=0ByQS_kT8kViSeEpDajIwOUFhaGc
    # gdown https://drive.google.com/uc?id=0ByQS_kT8kViSZnZPY1dmaHJzMHc
    # gdown https://drive.google.com/uc?id=0ByQS_kT8kViSb0VjVDJ3am40VVE
    # gdown https://drive.google.com/uc?id=0ByQS_kT8kViSTmQtd1VfWWFyUHM
    # gdown https://drive.google.com/uc?id=0ByQS_kT8kViSQ1ozYmlITXlUaDQ
    # gdown https://drive.google.com/uc?id=0ByQS_kT8kViSTVY1MnFGV0JiVkk
    # gdown https://drive.google.com/uc?id=0ByQS_kT8kViSYmhmbnp6d2I4a2M
    # gdown https://drive.google.com/uc?id=0ByQS_kT8kViSZl9aNGVuX0llcEU
    # gdown https://drive.google.com/uc?id=0ByQS_kT8kViSWXJ3R3hsZllsNVk
    # gdown https://drive.google.com/uc?id=0ByQS_kT8kViSR2N4cFpweURhTjg
    # gdown https://drive.google.com/uc?id=0ByQS_kT8kViScllKWnlaVU53Skk
    # gdown https://drive.google.com/uc?id=0ByQS_kT8kViSV2QxZW1rVXcxT1U
    # gdown https://drive.google.com/uc?id=0ByQS_kT8kViSNGNPTEFhdGxkMnM

    # Upload the resized images file link
}

echo "[INFO] Starting to download text-annotation files..."
download_text_annotation
echo "[INFO] Starting to download images files..."
download_images_annotation
#echo $(pwd)
