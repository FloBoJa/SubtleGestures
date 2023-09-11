# SubtleGestures

SubtleGestures is a project exploring gesture recognition using smartglasses. In this project, we used on J!ns Meme smartglasses with EOG and IMU sensors. This repository contains the implemented code, as well as our paper in which you can find our in-depth analysis, and results of our research project.

## Overview

SubtleGestures delves into the potential of leveraging subtle head and eye movements for intuitive interactions with smart devices. By utilising sensors of J!ns Meme smartglasses, our goal is to develop a robust gesture recognition system that seamlessly integrates into a variety of contexts. Through careful data collection, a modern neural network architecture, and real-time classification, we achieve impressive results in accurately recognizing gestures in real-world scenarios.

## Table of Contents

- [Key Project Components](#key-project-components)
- [Getting Started](#getting-started)
- [License](#license)

## Key Project Components

Here are the fundamental aspects that make up the SubtleGestures project:

- **Gesture Exploration:** Investigate a range of gestures designed to provide natural and discreet interactions.
- **Data Collection Tool:** Learn about the custom tool we've developed for efficient data gathering and precise gesture labeling.
- **Neural Network Architecture:** Gain insights into our adoption of LSTM-based neural networks for effective gesture recognition.
- **Performance Comparison:** Evaluate our model's accuracy against alternative classifiers, highlighting its strengths.
- **Complex Gesture Handling:** Discover our approach to addressing challenges posed by intricate gestures, including those involving the nose.
- **Real-Time Classifier:** Explore the real-time gesture classifier that enhances user interactions and enables real-world use.

## Getting Started

To use this project, follow these steps:

1. **Repository Cloning:** Obtain the project's source code by cloning the repository to your local environment.
2. **Dependencies Installation:** Set up required libraries using the `pip install -r requirements.txt` command.
3. **Data Collection and Labeling:** Follow data guidelines for the J!ns Meme Data Logger. Feel free to use `record_labels.py` and `split_data.py` for efficient data collection and precise labeling.
4. **Data Splitting (Optional):** If desired, use the `train_validate_test_split.py` script to split the dataset into training, validation, and test sets. This step is best-practice to evaluate the model's performance on unseen data and avoid overfitting.
5. **Neural Network Training:** Utilize provided scripts (`network.py` and `other_classifier.py`) to train and evaluate various neural network architectures and assess gesture recognition accuracy.
5. **Real-Time Classification:** After training the neural network, feel free to explore the real-time gesture classifier included in `network.py`.

## License

This project is licensed under the MIT License - see the [LICENSE](/LICENSE.txt) file for details.
