# Swadeshi-ASR

Automatic speech recognition (ASR) is an independent, machine-based process of decoding and transcribing oral speech. A typical ASR system receives acoustic input from a speaker through a microphone, analyzes it using a model or algorithm, and produces an output, usually in text. ASRs can be built using various approaches and this code
is based on the End to End Deep Learning approach.The end-to-end model uses deep learning to build a single model to directly map audio to characters or words.
It replaces the engineering process with the learning process and needs no domain expertise, so the end-to-end model is simpler for constructing and training.

Below is the general working pipeline of the End to End model:






![endtoend](https://user-images.githubusercontent.com/67675851/181423078-36561b66-3a1d-48cf-82fe-eeda57f22a4e.png)

The input is the raw speech or a 1 dimensional vector. The Encoder will extract the features from the speech input and it is converted into a feature vector. Most
commonly, the Convolutional Neural Networks are used to extract the features from signals. The aligner will align the feature vector to the sequence. The Decoder
will process the aligned feature sequence to generate the output sequence.

Swadeshi series will consist of ASRs in various Indian languages by using Indian languages datasets for the training of the models. 
Demo notebooks will be displayed in the notebooks folder of this repository.
