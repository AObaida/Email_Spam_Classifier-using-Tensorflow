# Email_Spam_Classifier-using-Tensorflow

# Project Overview
A machine learning project focused on building an Email Spam Classifier to accurately predict if an incoming email is "spam" or "ham" (not spam).
The model processes raw text data from the spam_ham_dataset.csv file.
The core classifier uses a deep learning approach implemented with TensorFlow and Keras.

# Key Features & Techniques
Deep Learning Classification: Implemented a model using the TensorFlow/Keras for robust text classification.
Text Preprocessing: Utilizes the NLTK library for text cleaning, including the removal of standard English stopwords and general text normalization.
Tokenization & Sequencing: Applied Keras's Tokenizer and pad_sequences utilities to prepare the email text for input into the neural network.
Exploratory Data Analysis (EDA): Includes visualizations (using Seaborn and Matplotlib) and Word Clouds to analyze the distribution and key terms in spam and non-spam emails.
Model Optimization: Employed callbacks such as EarlyStopping and ReduceLROnPlateau to prevent overfitting and optimize the training process.

# Technologies & Dependencies
Language: Python 3
Deep Learning: tensorflow / keras
Data Handling: pandas, numpy
Natural Language Processing (NLP): nltk (for stopwords)
Visualization: matplotlib, seaborn, wordcloud
Utilities: sklearn (for train_test_split)
