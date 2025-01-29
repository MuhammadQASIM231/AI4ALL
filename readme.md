# Fake and Real News Classification

## Overview
This project focuses on detecting fake and real news using a dataset containing labeled news articles. The dataset includes both true and fake news articles, which are processed, cleaned, and visualized to understand patterns before classification.

## Contributors
This project was written by **Muhammad Qasim** with contributions from **Elaine Gombos**.

## Dataset
The dataset used in this project is sourced from Kaggle and consists of two files:
- `Fake.csv` (containing fake news articles)
- `True.csv` (containing true news articles)

## Steps Involved

### 1. Importing Required Libraries
The code starts by importing necessary Python libraries such as:
- `nltk` for text processing
- `numpy` and `pandas` for data handling
- `seaborn` and `matplotlib` for visualization

```python
import nltk
nltk.download('wordnet')
nltk.download('stopwords')
import numpy as np 
import pandas as pd 
import os
```

### 2. Loading the Dataset
The news articles are loaded into pandas DataFrames and labeled with a category column:
```python
fake = pd.read_csv("../input/fake-and-real-news-dataset/Fake.csv")
true = pd.read_csv("../input/fake-and-real-news-dataset/True.csv")
fake["category"] = 1
true["category"] = 0
df = pd.concat([fake, true]).reset_index(drop=True)
```

### 3. Data Visualization
Count plots are created to visualize the distribution of fake and real news, along with different news subjects.
```python
import seaborn as sns
import matplotlib.pyplot as plt
sns.countplot(x="category", data=df)
plt.title("Count of Fake and True News")
```

### 4. Data Cleaning
- Missing values are checked and handled.
- Empty text fields are removed by merging the title with the text.
- Stopwords from NLTK and spaCy are combined.
- The text is cleaned by removing special characters, simplifying contractions, and lemmatizing words.

```python
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer 
import spacy
import re
nlp = spacy.load("en_core_web_sm")
lemma = WordNetLemmatizer()
Stopwords = set(nlp.Defaults.stop_words) | set(stopwords.words('english'))
```

A function is implemented to clean and preprocess the text:
```python
def clean_text(text):
    text = text.lower()
    text = re.sub(r"i'm", "i am", text)
    text = re.sub(r"[^A-Za-z0-9]+", ' ', text)
    string = " ".join([lemma.lemmatize(word) for word in text.split() if word not in Stopwords])
    return string
```

## Conclusion
This project effectively classifies fake and real news articles using a structured preprocessing pipeline. The contributions of **Elaine Gombos** helped improve the implementation and data handling processes.

## Future Work
- Implement machine learning models for classification.
- Use deep learning approaches such as LSTMs or transformers.
- Explore different feature engineering techniques to improve accuracy.

---
This README provides an overview of the dataset, data processing steps, and future enhancements for this project. Feel free to explore and contribute further!
