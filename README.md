# A Personality-Labeled Semantic Dataset from Facial Expressions, Gaze, and Head Movement Cues

# Overview
The software takes as input pre-extracted facial Action Units (AUs), eye gaze directions, and head rotation parameters and trains a deep neural network with attention and convolutional layers to predict Five-Factor personality trait labels. It supports retraining with updated annotations and includes correlation analysis of feature contributions.


## Getting Started

These instructions will get you a copy of the project up and running on your local machine for development and testing. The dataset can be generated using two models provided in the project.

### Prerequisites

What things you need to install and how to install them:

- Python 3.8+

### Installation

A step-by-step series of examples that tell you how to get a development environment running:



 **Clone the repository:**
   ```bash
   git clone https://github.com/kodavatiSrikar/Personality-of-Head-Motion.git
   cd Dataset-for-facial-expression-of-personality
   ```

Install the requirements using the package manager [pip](https://pip.pypa.io/en/stable/).

```bash
pip install -r requirements.txt
```
## Downloading Files from Google Drive

To download the necessary files from Google Drive, follow these steps:

1. Copy the file's sharing link from Google Drive.
   [Dataset](https://drive.google.com/drive/folders/15HHCb6eOnz4kK3AmFgACZvbNZY89oPSC?usp=sharing)
2. Download the folder, unzip the files inside, and copy the files inside the Personality-of-Head-Motion folder.

## Usage

Follow the below sections to train and run the inference of the models. 

## [Note]

Inference of the model can be executed without training the model. The pre-trained weights for the  model is provided in Google Drive.

## Data Augmentation
Augment the training data(data_combined.csv)
```bash
python data_augmentation.py
```

## Hybrid Model

## Training

Run the following in the project directory to train the hybrid model

```bash
python attn_regression.py
```


## Retraining

## Data Augmentation
Augment the retraining data(retrain_data.csv)
```bash
python data_augmentation.py
```
Retraining the hybrid model with the data obtained from the user study. This step loads the pretrained weights(au_180.pt) as a base model.

```bash
python attn_retrain.py
```
## Testing

Run the following in the project directory to test the hybrid model performance.

```bash
python attn_test.py
```

## Deployment

Run the following in the project directory to generate the personality traits data using a hybrid model. The deployment uses trains model weights(3iternorm_180.pt) to run the inference.

```bash
python attn_deploy.py
```

## Correlation analysis

Replicate the analysis (Figure 6 in the paper) by computing Spearman correlations between mean AU intensities and personality traits:

```bash
python au_analysis.py
```

## Action Unit, Head, and Gaze Parameter Extraction

Action Unit, Head, and Gaze can be extracted from custom videos using the OpenFace library, which employs the FACS principle. Please use the following [Documention](https://github.com/TadasBaltrusaitis/OpenFace/wiki) to obtain the Action Unit, Head, and Gaze used to input our model.


