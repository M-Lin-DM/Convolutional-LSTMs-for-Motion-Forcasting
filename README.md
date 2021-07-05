# Convolutional-LSTMs-for-Motion-Forcasting
Keras-tensorflow code for training a frame-by-frame binary classifier with video input + code for computing targets.

Please see the blog posts for a detailed description of this project.
[https://m-lin-dm.github.io/ConvLSTM_Fish_1/](https://m-lin-dm.github.io/ConvLSTM_Fish_1/)

YouTube video presenting my results:

**Project In Brief - TLDR** 
- Under my laboratory conditions, *Serpae tetra* fish exhibit aperiodic, "bursty" movement patterns---typically, one fish begins chasing another when the two get too close together. [[video]](link)
- **I hypothesized that**  ConvLSTM would be particularly well-suited to model these dynamics. The underlying assumption is that the fishes' prior dynamics encodes some information on if/when a burst is likely to occur.
<!-- - To obtain ground truth labels, I converted this problem into a *self-supervised* setting. Roughly, each frame $$i$$ was algorithmically labeled as 1 if the group mean speed $$\tilde{y_i}$$ was low and frame $$i$$ preceded a sufficiently large increase in $$\tilde{y_i}$$ across frames $$[i, i+45]$$. Conversely, frame $$i$$ was labeled as 0 if it did not precede a large increase in speed. The method of computing $$\tilde{y_i}$$ and the full details of the labeling algorithm can be found in [part 3](/ConvLSTM_Fish_3/) -->
- **Research Goal: Train a ConvLSTM-based model that can take a video clip of fish as input and predict whether or not a burst of motion will occur during the subsequent 1.5 seconds.** (Visually, this burst corresponds to sudden aggressive interactions between two or more fish and a spike in group mean speed.)
- **Methods:** I first filmed a group of 6 *Serpae tetra* fish from overhead, for 30 mins. I converted this into an image dataset containing ~54000 grayscale frames with 120x160x1 resolution. I trained a ConvLSTM-based model to act as a **frame-by-frame binary classifier**. This recurrent neural network takes as input a 4 second clip of video frames (stacked as a single 4D tensor of shape (time-steps x width x height x channels)) and outputs the probability that a 'spike' in group mean velocity will occur during the next 1.5 seconds (45 frames).
- **Result:** The model does fairly well at identifying frames that come right before the fish chase each other. It achieves an ROC-AUC of 0.83 and a peak accuracy of 84.7%, matching the algorithmic, baseline classifier (AUC: 0.84 Peak accuracy: 85.4%). Using a probability threshold that maximizes Youden's statistic, the model achieved a true positive rate of 82.9%, with a false positive rate of 31.0%. 
<!-- This helps confirm that the fishes' prior spatial configurations and dynamics contain information that is useful in predicting the onset of aggressive interactions. -->
- This 5-part series is a hybrid between an academic paper and an informal guide to doing a deep learning research project. I will
1. Introduce the problem we're trying to solve and explain why deep learning/ConvLSTMs constitute a novel approach to it. (This post) 
2. Report my methods and findings in detail
3. Explain key parts of my python-keras-tensorflow code.
4. Provide a step-by-step guide outlining the subtasks required to take a project like this from start to finish.  
