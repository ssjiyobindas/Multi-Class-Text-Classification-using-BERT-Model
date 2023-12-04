# NLP Project for Multi-Class Text Classification using BERT Model

## Project Overview

 This project dives into advanced techniques for multiclass text classification. In this project, we harness the power of BERT (Bidirectional Encoder Representations) - an open-source ML framework by Google, renowned for delivering state-of-the-art results in various NLP tasks.

## Business Overview

In our journey through NLP algorithms, from Naïve Bayes to RNN and LSTM, we now embark on the efficiency of BERT for text classification. BERT's bidirectional encoding and pre-trained capabilities elevate our project to new heights of accuracy and performance.

## Aim

Our goal is to leverage the pre-trained BERT model for multiclass text classification, utilizing a dataset containing over two million customer complaints about consumer financial products.

## Data Description

The dataset includes customer complaints with corresponding product categories. Our objective is to predict product categories based on the text of the complaints.

## Tech Stack

- **Language:** Python
- **Libraries:** pandas, torch, nltk, numpy, pickle, re, tqdm, sklearn, transformers

## Prerequisite

Before diving in, ensure you have the required packages installed. Refer to the `requirements.txt` file for the specific libraries and versions needed.

## Approach

1. **Installing Necessary Packages:**
   - Use the pip command to install required packages.

2. **Importing Required Libraries:**
   - Set the stage by importing essential libraries.

3. **Defining Configuration File Paths:**
   - Establish paths for configuration files.

4. **Processing Text Data:**
   - Read and preprocess the CSV file.
   - Handle null values, duplicate labels, and encode labels.

5. **Data Preprocessing:**
   - Convert text to lowercase.
   - Remove punctuation, digits, consecutive instances of 'x', and extra spaces.
   - Tokenize the text and save tokens.

6. **Model Building:**
   - Create the BERT model.
   - Define PyTorch dataset functions.
   - Implement functions for model training and testing.

7. **Train BERT Model:**
   - Load files and split data.
   - Create PyTorch datasets and data loaders.
   - Define the model, loss function, and optimizer.
   - Train the model and test its performance.

8. **Predictions of New Text:**
   - Make predictions on new text data using the trained model.

## Modular Code Overview

Upon unzipping the `modular_code.zip` file, you'll find folders:

1. **Input:**
   - Contains the analysis data, in this case, `complaints.csv`.

2. **Output:**
   - Contains essential files for model training:
     - `bert_pre_trained.pth`
     - `label_encoder.pkl`
     - `labels.pkl`
     - `tokens.pkl`

3. **Source:**
   - Holds modularized code in Python files for better organization:
     - `model.py`
     - `data.py`
     - `utils.py`

4. **Config:**
   - `config.py` file with project configurations.

5. **Engine:**
   - `Engine.py` is the main file for running the entire code, training the model, and saving it in the output folder.

6. **Notebook:**
   - `bert.ipynb` is the original notebook used in the development.

7. **Processing and Predictions:**
   - `processing.py` processes the data.
   - `predict.py` makes predictions on the data.

8. **README and Requirements:**
   - `README.md` provides detailed instructions, and `requirements.txt` lists necessary libraries.

## Project Takeaways

1. **Understanding the Business Problem:**
   - Grasping the intricacies of multiclass text classification.

2. **Exploring Pre-trained Models:**
   - Introduction to the concept and significance of pre-trained models.

3. **BERT Model Insights:**
   - Understanding the architecture and functioning of BERT.

4. **Data Preparation Techniques:**
   - Handling spaces, digits, and punctuation for effective model input.

5. **BERT Tokenization:**
   - Implementing BERT tokenization for text processing.

6. **Model Architecture and Training:**
   - Creating and training the BERT model using CUDA or CPU.

7. **Predictions on New Text Data:**
   - Applying the trained model for predictions on unseen text data.

## Additional Information

Feel free to explore the `modular_code.zip` for organized code snippets. The project provides a seamless experience with pre-trained models, ensuring quick and efficient use without the need to retrain from scratch.

For a more hands-on experience, refer to the `bert.ipynb` notebook and follow the instructions in the `README.md` file for detailed guidance.

Happy coding! 🚀✨
