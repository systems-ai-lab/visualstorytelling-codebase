# Building Vocabulary from the Stories of Images-in-Sequence Text Annotation
Before we start learning a new model from the VIST dataset, we need to prepare the words vocabulary that used in the overall story. Building vocabulary is not included in the learning process but still related to the overall learning process. As we know, the computer machine can only process numerical data, especially we use a tensor representation containing only float numbers. This process attempt to generate not only word vocabulary but also its integer index and frequency for each word.

| Index        | Word           | Frequency |
| :-------------: |:------------|:------------:|
|1|word-1|10|
|2|word-2|12|
|...|...|...|
|n|word-n|20|

### What is ```n```? 
Shown in the table above, it contains word-index pair with its frequency, ```n``` representing the number of words contained in the vocabulary. This may vary depends on the value of the minimum threshold of word appearances in the story. A word will not be included in the vocabulary if it appears less than the minimum threshold frequency. 

### Adding special ```<token>``` in vocabulary
Special token is needed for several purposes, in this document we will show 4 special token as follows:

| No.        | Special Token           | Purpose |
| :-------------: |:------------|:------------|
|1|```<start>```|This token indicates the beginning of the sentence. It will appended for every sentences in story.|
|2|```<end>```|In contrary with ```<start>``` token, this token indicates the end of the sentence in story. This token appended in as the last word of story|
|3|```<unk>```|This token is used for replacing the a word that not included in vocabulary due to insufficient of minimum threshold|

To be continued...

## References
1. https://github.com/tkim-snu/GLACNet
2. https://github.com/yunjey/pytorch-tutorial/tree/master/tutorials/03-advanced/image_captioning
