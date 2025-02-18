"""
process.py

A module for preprocessing datasets with common data cleaning, transformation, and text preprocessing functions.

Author: Nguyen Quang Phu
Date: 2025-01-21
"""

import pandas as pd
import numpy as np
import re
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.tokenize import sent_tokenize
from sklearn.feature_extraction.text import TfidfVectorizer

def handle_missing_values(data, strategy="mean", fill_value=None, columns=None):
    """
    Handle missing values in the dataset.
    
    :param data: pandas DataFrame
    :param strategy: Strategy to fill missing values. Options: 'mean', 'median', 'mode', 'constant'.
    :param fill_value: Value to use if strategy is 'constant'.
    :param columns: List of columns to process. If None, processes all columns.
    :return: DataFrame with missing values handled.
    """
    columns = columns or data.columns
    if strategy == "mean":
        return data.fillna({col: data[col].mean() for col in columns})
    elif strategy == "median":
        return data.fillna({col: data[col].median() for col in columns})
    elif strategy == "mode":
        return data.fillna({col: data[col].mode()[0] for col in columns})
    elif strategy == "constant":
        if fill_value is None:
            raise ValueError("`fill_value` must be specified when strategy is 'constant'.")
        return data.fillna({col: fill_value for col in columns})
    else:
        raise ValueError(f"Unknown strategy: {strategy}")

def encode_categorical_columns(data, encoding_type="one_hot", columns=None):
    """
    Encode categorical columns in the dataset.
    
    :param data: pandas DataFrame
    :param encoding_type: Encoding type. Options: 'one_hot', 'label'.
    :param columns: List of columns to encode. If None, encodes all object columns.
    :return: DataFrame with encoded categorical columns.
    """
    columns = columns or data.select_dtypes(include=["object", "category"]).columns
    if encoding_type == "one_hot":
        return pd.get_dummies(data, columns=columns, drop_first=True)
    elif encoding_type == "label":
        from sklearn.preprocessing import LabelEncoder
        encoder = LabelEncoder()
        for col in columns:
            data[col] = encoder.fit_transform(data[col])
        return data
    else:
        raise ValueError(f"Unknown encoding type: {encoding_type}")

def normalize_columns(data, columns=None):
    """
    Normalize numerical columns to the range [0, 1].
    
    :param data: pandas DataFrame
    :param columns: List of columns to normalize. If None, normalizes all numeric columns.
    :return: DataFrame with normalized columns.
    """
    columns = columns or data.select_dtypes(include=["number"]).columns
    data[columns] = data[columns].apply(lambda x: (x - x.min()) / (x.max() - x.min()), axis=0)
    return data

def standardize_columns(data, columns=None):
    """
    Standardize numerical columns to have mean 0 and standard deviation 1.
    
    :param data: pandas DataFrame
    :param columns: List of columns to standardize. If None, standardizes all numeric columns.
    :return: DataFrame with standardized columns.
    """
    columns = columns or data.select_dtypes(include=["number"]).columns
    data[columns] = data[columns].apply(lambda x: (x - x.mean()) / x.std(), axis=0)
    return data

def remove_outliers(data, method="z_score", threshold=3, columns=None):
    """
    Remove outliers from the dataset.
    
    :param data: pandas DataFrame
    :param method: Method to detect outliers. Options: 'z_score', 'iqr'.
    :param threshold: Threshold for detecting outliers (z-score or IQR multiplier).
    :param columns: List of columns to process. If None, processes all numeric columns.
    :return: DataFrame with outliers removed.
    """
    columns = columns or data.select_dtypes(include=["number"]).columns
    if method == "z_score":
        from scipy.stats import zscore
        z_scores = data[columns].apply(zscore)
        return data[(z_scores.abs() <= threshold).all(axis=1)]
    elif method == "iqr":
        for col in columns:
            q1 = data[col].quantile(0.25)
            q3 = data[col].quantile(0.75)
            iqr = q3 - q1
            lower_bound = q1 - threshold * iqr
            upper_bound = q3 + threshold * iqr
            data = data[(data[col] >= lower_bound) & (data[col] <= upper_bound)]
        return data
    else:
        raise ValueError(f"Unknown method: {method}")

def drop_duplicates(data):
    """
    Drop duplicate rows from the dataset.
    
    :param data: pandas DataFrame
    :return: DataFrame without duplicate rows.
    """
    return data.drop_duplicates()

def remove_special_characters(text):
    """
    Remove special characters and punctuation from text.
    """
    text = re.sub(r"[^a-zA-Z0-9\s]", "", text)
    text = re.sub(r'http\S+|https?://\S+|www\.\S+', '', text)
    text = re.sub(r'\[.*?\]', '', text)
    text = re.sub(r'<.*?>+', '', text)
    text = re.sub(r'\n|\t|\r|\f|\b', '', text)
    text = re.sub(r'\w*\d\w*', '', text)
    return text.strip()

def lowercase_text(text):
    """
    Convert text to lowercase.
    """
    return text.lower()

def tokenize_text(text):
    """
    Tokenize text into words.
    """
    return word_tokenize(text)

def remove_stopwords(words, language="english"):
    """
    Remove stopwords from a list of words.
    """
    stop_words = set(stopwords.words(language))
    return [word for word in words if word not in stop_words]

def stem_words(words):
    """
    Apply stemming to a list of words.
    Stemming reduces words to their base form using rule-based heuristics.
    """
    from nltk.stem import PorterStemmer
    stemmer = PorterStemmer()
    return [stemmer.stem(word) for word in words]

def lemmatize_words(words):
    """
    Apply lemmatization to a list of words.
    Lemmatization reduces words to their dictionary form using linguistic rules.
    """
    from nltk.stem import WordNetLemmatizer
    lemmatizer = WordNetLemmatizer()
    return [lemmatizer.lemmatize(word) for word in words]

def remove_html_tags(text):
    """ 
    Remove html tags from text
    """
    clean_text = re.sub(r'<.*?>', '', text)
    return clean_text

def remove_mention(text):
    """ 
    Remove @mentions
    """
    # Remove @mentions
    clean_text = re.sub(r'@\w+', '', text)
    return clean_text

def remove_urls(text):
    """ 
    Remove urls from text
    """
    clean_text = re.sub(r'http\S+', '', text)
    return clean_text

def replace_chat_words(text):
    chat_words = {
        "BRB": "Be right back",
        "BTW": "By the way",
        "OMG": "Oh my God/goodness",
        "TTYL": "Talk to you later",
        "OMW": "On my way",
        "SMH/SMDH": "Shaking my head/shaking my darn head",
        "LOL": "Laugh out loud",
        "TBD": "To be determined", 
        "IMHO/IMO": "In my humble opinion",
        "HMU": "Hit me up",
        "IIRC": "If I remember correctly",
        "LMK": "Let me know", 
        "OG": "Original gangsters (used for old friends)",
        "FTW": "For the win", 
        "NVM": "Nevermind",
        "OOTD": "Outfit of the day", 
        "Ngl": "Not gonna lie",
        "Rq": "real quick", 
        "Iykyk": "If you know, you know",
        "Ong": "On god (I swear)", 
        "YAAAS": "Yes!", 
        "Brt": "Be right there",
        "Sm": "So much",
        "Ig": "I guess",
        "Wya": "Where you at",
        "Istg": "I swear to god",
        "Hbu": "How about you",
        "Atm": "At the moment",
        "Asap": "As soon as possible",
        "Fyi": "For your information",
        "Tbh": "To be honest",
        "Wtf": "What the fuck",
        "Idk": "I don't know"
    }
    
    for word, expanded_form in chat_words.items():
        text = text.replace(word, expanded_form)
    return text

def text_preprocessing(
    text,
    remove_special=True,
    to_lowercase=True,
    remove_stopwords_flag=True,
    stem_flag=False,
    lemmatize_flag=False,
    tokenize_flag=True,
    language="english",
):
    """
    Perform text preprocessing with multiple configurable steps.

    :param text: The input text string to preprocess.
    :param remove_special: Whether to remove special characters.
    :param to_lowercase: Whether to convert the text to lowercase.
    :param remove_stopwords_flag: Whether to remove stopwords from the text.
    :param stem_flag: Whether to apply stemming.
    :param lemmatize_flag: Whether to apply lemmatization.
    :param tokenize_flag: Whether to tokenize the text into words.
    :param language: Language to use for stopwords and tokenization.
    :return: Preprocessed text.
    """
    from nltk.tokenize import sent_tokenize
    
    # Split the text into sentences for better handling of long texts
    sentences = sent_tokenize(text)

    processed_sentences = []
    for sentence in sentences:
        # Step 1: Remove special characters
        sentence = remove_mention(sentence)
        sentence = remove_html_tags(sentence)
        sentence = remove_urls(sentence)
        sentence = replace_chat_words(sentence)
        
        if remove_special:
            sentence = remove_special_characters(sentence)
        
        # Step 2: Convert to lowercase
        if to_lowercase:
            sentence = lowercase_text(sentence)
        
        # Step 3: Tokenize the sentence into words
        words = tokenize_text(sentence)
        
        # Step 4: Remove stopwords
        if remove_stopwords_flag:
            words = remove_stopwords(words, language)
        
        # Step 5: Apply stemming
        if stem_flag:
            words = stem_words(words)
        
        # Step 6: Apply lemmatization
        if lemmatize_flag:
            words = lemmatize_words(words)
        
        # Step 7: Reconstruct the sentence from words
        processed_sentences.append(" ".join(words))

    # Merge sentences back into text
    return " ".join(processed_sentences)

# if __name__ == "__main__":
#     # Example text
#     text = """
#     This is an example sentence to demonstrate text preprocessing!
#     It includes punctuation, CAPITALIZED words, and stopwords.
#     """

#     processed_text = text_preprocessing(
#         text,
#         remove_special=True,
#         to_lowercase=True,
#         remove_stopwords_flag=False,
#         stem_flag=False,
#         lemmatize_flag=True,
#     )

#     print("Original Text:\n", text)
#     print("Processed Text:\n", processed_text)
    