## Introduction
    This project is on CNN with keras to process audio files and predict spoken words in it.
    To extract a feature vector containing all information about the linguistic message, MFCC mimics some parts of the human speech production and speech perception. 
 ## MFCC
    MFCC mimics the logarithmic perception of loudness and pitch of human auditory system and tries to eliminate speaker dependent characteristics by excluding the fundamental frequency and their harmonics. 
    To represent the dynamic nature of speech the MFCC also includes the change of the feature vector over time as part of the feature vector [3,4].
## CNN
  A Convolutional Neural Network (CNN) is comprised of one or more convolutional layers (often with a subsampling step) and then followed by one or more fully connected layers as in a standard multilayer neural network. 
  The architecture of a CNN is designed to take advantage of the 2D structure of an input image (or other 2D input such as a speech signal). 
  This is achieved with local connections and tied weights followed by some form of pooling which results in translation invariant features. 
  Another benefit of CNNs is that they are easier to train and have many fewer parameters than fully connected networks with the same number of hidden units. 
  In this article we will discuss the architecture of a CNN and the back propagation algorithm to compute the gradient with respect to the parameters of the model in order to use gradient based optimization. 
  See the respective tutorials on convolution and pooling for more details on those specific operations.
  
### uses:
  python3 preprcess.py
  python3 train.py
