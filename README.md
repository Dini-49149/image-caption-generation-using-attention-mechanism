# image-caption-generation-using-attention-mechanism

The research was carrying out to improve the accuracy of the caption generating from an image

## Introduction:

Image caption Generator is a popular research area of Artificial Intelligence that deals with image understanding and a language description for that image. Generating well-formed sentences requires both syntactic and semantic understanding of the language.Being able to describe the content of an image using accurately formed sentences is a very challenging task, but it could also have a great impact, by helping visually impaired people better understand the content of images. 

This task is significantly harder in comparison to the image classification or object recognition tasks that have been well researched. 

The biggest challenge is most definitely being able to create a description that must capture not only the objects contained in an image, but also express how these objects relate to each other.


## Prerequisites to get started:
* Python programming
* Tensorflow and Keras
* Convolutional Neural Networks and its implementation
* RNN and LSTM 
* Transfer learning 
* Encoder and Decoder Architectures

## Problem with classic image captioning model:

when the model is trying to generate the next word of the caption, this word is usually describing only a part of the image. It is unable to capture the essence of the entire input image. Using the whole representation of the image h to condition the generation of each word cannot efficiently produce different words for different parts of the image.This is exactly where an **Attention mechanism** is helpful.

This ability of self-selection is called attention. The attention mechanism allows the neural network to have the ability to focus on its subset of inputs to select specific features. Attention mechanism has been a go-to methodology for practitioners in the Deep Learning community. It was originally designed in the context of Neural Machine Translation using Seq2Seq Models, but today we’ll take a look at its implementation in Image Captioning.

Attention mechanism has been a go-to methodology for practitioners in the Deep Learning community. It was originally designed in the context of Neural Machine Translation using Seq2Seq Models, but today we’ll take a look at its implementation in Image Captioning.

## Approach to the problem statement:

The encoder-decoder image captioning system would encode the image, using a pre-trained Convolutional Neural Network that would produce a hidden state. Then, it would decode this hidden state by using an LSTM and generate a caption. We use Inception v3 which is pretrained on ImageNet dataset and Spacy or Glove embeddings  to encode images and text captions provided. 

For each sequence element, outputs from previous elements are used as inputs, in combination with new sequence data. This gives the RNN networks a sort of memory which might make captions more informative and contextaware. 
But RNNs tend to be computationally expensive to train and evaluate, so in practice, memory is limited to just a few elements. Attention models can help address this problem by selecting the most relevant elements from an input image. 
With an Attention mechanism, the image is first divided into n parts, and we compute an image representation of each When the RNN is generating a new word, the attention mechanism is focusing on the relevant part of the image, so the decoder only uses specific parts of the image.

## Concept of Attention mechanism:

With the attention mechanism,  the image is first divided into n parts, and we compute with a Convolutional Neural Network (CNN) representations of each part h1,…, hn. When the RNN is generating a new word, the attention mechanism is focusing on the relevant part of the image, so the decoder only uses specific parts of the image.
In Bahdanau or Local attention, attention is placed only on a few source positions. As Global attention focuses on all source side words for all target words, it is computationally very expensive. To overcome this deficiency local attention chooses to focus only on a small subset of the hidden states of the encoder per target word.
Local attention first finds an alignment position and then calculates the attention weight in the left and right windows where its position is located and finally weights the context vector. The main advantage of local attention is to reduce the cost of the attention mechanism calculation.

## Understanding the Dataset:

There are many open source datasets available for this problem, like Flickr 8k (containing8k images), Flickr 30k (containing 30k images), MS COCO (containing 180k images), etc. I have used the Flickr8k dataset in which each image is associated with five different captions that describe the entities and events depicted in the image that were collected. 
Flickr8k is a good starting dataset as it is small in size and can be trained easily on low-end laptops/desktops using a CPU.
This dataset contains 8000 images each with 5 captions (as we have already seen in the Introduction section that an image can have multiple captions, all being relevant simultaneously).
These images structured as follows:
* Flick8k/
    * Flick8k_Dataset/ :- contains the 8000 images
    * Training set---------- 6000 images
    * Dev set       ----------- 1000 images
    * Test set      ----------- 1000 images
* Flick8k_Text/
    * Flickr8k.token.txt:- contains the image id along with the 5 captions
 
## Objective of the research:
 
1. Finding out the labels of object component where the attention mechanism is focusing.
2. There should be a collection of all image vectors which are there in the training set , so that test image vector can find similar image vector in the training images using cosine similarity, whichever vector is most similar, utilise its captions to find the similarity of the object labels obtained from the attention part.
3. If the caption words are similar to the labels then that label has to be used in the decoder part.
 
## Evaluation metric:
We use the BLEU measure to evaluate the result of the the test set generated captions. The BLEU is simply taking the fraction of n-grams in the predicted sentence that appears in the ground-truth.
BLEU is a well-acknowledged metric to measure the similarly of one hypothesis sentence to multiple reference sentences. Given a single hypothesis sentence and multiple reference sentences, it returns value between 0 and 1. The metric close to 1 means that the two are very similar.

**Model Description:** Once we finish the code

**Results:** yet to be obtained 

**Conclusion:** yet to be derived
 
 
	
 

