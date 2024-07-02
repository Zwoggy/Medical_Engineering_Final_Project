# Sleepy Driver EEG Brainwave Data Project

This project is designed to build and evaluate a machine learning model capable of detecting whether a driver is sleepy based on Electroencephalography (EEG) brainwave data. 

## Data Collection

EEG signal data was collected from four different drivers while they were awake and asleep. To ensure safety, the data was collected in a controlled environment where the drivers were not actually driving. Each driver wore the [NeuroSky MindWave](http://neurosky.com) sensor headset for approximately 5-8 minutes per state (awake/asleep), and the signals were captured roughly every second. EEG signals are quantified in units of microvolts squared per hertz (μV²/Hz).

## Data Description

The EEG signals were collected from a single channel positioning on the drivers' forehead using the NeuroSky MindWave sensor headset. The use of single-channel EEG monitoring could account for the high signal values as clinical-grade EEG devices typically utilize multi-channel configurations. 

Additional attributes such as "Attention" and "Meditation" levels, calculated directly by the headset, are also available within the data. However, as these are headset-specific calculations, they might present varying levels of reliability across different individuals and scenarios. 

The dataset used for this project can be accessed via following link: [Download the Dataset](https://elearning.th-wildau.de/pluginfile.php/570787/block_quickmail/attachments/31610/1716540497_attachments.zip?forcedownload=1)

## Project Structure

In the main python scripts, data from the EEG signals are first visualized to show their correlation and relation with the target variable (Classifications: Sleepy/Awake). Various machine learning classifiers are implemented and their performance is measured. Various preprocessing steps such as Label Encoding, train-test split, Data Resampling, Data Imputation, and Scaling are performed as part of training and testing our classifiers. There is also a function to train a simple Deep Neural Network.

Main algorithms used:
1. Logistic Regression
2. RandomForestClassifier
3. SVC (Support Vector Machine Classifier)
4. k-Nearest Neighbors
5. Gradient Boosting Classifier

This repository includes:

**Python script** - A .py file or a link to a Git repository containing the code utilized to analyze the data and execute the machine learning models.

**Documentation** - A detailed README.md file which provides information associated with the project, its data, the methodology employed and results obtained.

**Requirements.txt File** - A requirements.txt file outlining the necessary Python packages required to successfully run the Python script.

## Contributions
Author: Florian Zwicker

