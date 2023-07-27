# Overview

This is a set of scripts and models for the real time transcription of semaphore. The scripts allow the creation of new labelled images, files of pose keypoints, and the training of a classifying NN. The system is based on a blaze pose keypoint extractor, with a simple three layer NN.

It was found that a simple ~1200 image dataset from three different sources generalises reasonably well to new image feeds.

There is a blog post [here](https://www.seoirse.net/posts/transcribing-semaphore) which contains detail about this model and example usage.

# Quickstart

The file process_poses.py will load a pretrained model (train_semaphore.pt) and apply it to a given videofeed or video feed.

# Contents

- label_new_data.py
    - Run through a video file with arrow keys. Creates labelled image files by clicking the keyboard key. This can be used by produce_train_array to extract keypoints to train the model.
- produce_train_array
    - Reads in labelled images and combines them to produce a training array. There are different naming conditins for the downloaded datasets, and my dataset.   
- train_network
    - Loads the labelled pose data and uses it to train a simple NN to categorise. This is then saved to file for use.
- test_model
    - Loads the saved model, and runs it against a csv of labelled pose data. This tests only the NN, and not the pose extractor. As is, it loads labelled data unseen during training.
- pose.py
    - A class which processes a given frame using blaze pose extractor
- process_poses
    - Runs the full system either on a live camera feed or on a video file.

# Examples

![[Hello](https://github.com/seoirsem/semaphore/blob/main/Images/charlotte_world.gif)

![[World](https://github.com/seoirsem/semaphore/blob/main/Images/seoirse_hello.gif)
