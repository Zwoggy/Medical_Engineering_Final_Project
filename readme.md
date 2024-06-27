# Sleepy Driver EEG Brainwave Data Project

This project aims to build a machine learning classifier for detecting whether a driver is sleeply by analyzing EEG brainwave data.

The dataset included in this project is consist of EEG signal data from 4 drivers while they were awake and asleep. For safety precautions, the signals are acquired while drivers were not actually driving. The drivers wore the [NeuroSky MindWave](http://neurosky.com) sensor for 5-8 minutes for each label (sleepy, not sleepy) and the signals were captured roughly every second. The signals are measured in units of microvolts squared per hertz (μV²/Hz).

## Data Description

These EEG signals were taken from a single location on each driver's forehead using the NeuroSky MindWave headset. The high signal values could be attributed to the single location method as opposed to medical-grade EEG devices that typically use multiple electrodes placed on different parts of the scalp.

It's important to note that there were no pre-processing done to the signals. Additional attributes like Attention and Meditation are calculated directly by the headset and somewhat considered to be unreliable.

For convenience, I have included a download link to the dataset: [Download the Dataset](https://elearning.th-wildau.de/pluginfile.php/570787/block_quickmail/attachments/31610/1716540497_attachments.zip?forcedownload=1)

## Project Structure

The final project deliverable will include:

1. Python script (.py) or a link to a Git repository with the coding documentation.
2. Documentation in the form of README.md following markdown syntaxes ([Markdown Basic Syntax](https://www.markdownguide.org/basic-syntax/))
3. A requirements.txt file for setting up the Python environment ([Python Environment Setup](https://frankcorso.dev/setting-up-python-environment-venv-requirements.html))

Additionally, the final presentation deliverable will include:

1. A brief explanation of the project task.
2. A description of the EEG method used and the basics of the EEG signal.
3. Details about the dataset and the experiment conducted.
4. Details about the machine learning model(s) used.
5. The results from the data analysis.
6. Conclusion
7. References
