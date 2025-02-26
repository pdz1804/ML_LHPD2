"""
process.py

A module for preprocessing datasets with common data cleaning, transformation, and text preprocessing functions.

Author: Nguyen Quang Phu
Last Modified: 2025-01-21

This module includes functions for:
- Removing special characters, punctuation, HTML tags, URLs, and @mentions from text.
- Converting text to lowercase.
- Tokenizing text into words.
- Removing stopwords from text.
- Applying stemming and lemmatization to words.
- Replacing common chat words with their expanded forms.
- Removing leading and trailing whitespace from text.
- Performing comprehensive text preprocessing with configurable steps.
"""

import pandas as pd
import numpy as np
import re
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.tokenize import sent_tokenize
from sklearn.feature_extraction.text import TfidfVectorizer
import string, time

def remove_special_characters(text):
    """
    Remove special characters and punctuation from text.

    This function performs the following steps:
    1. Remove all characters that are not alphanumeric or whitespace.
    2. Remove URLs starting with http, https, or www.
    3. Remove text within square brackets.
    4. Remove HTML tags.
    5. Remove newline, tab, carriage return, form feed, and backspace characters.
    6. Remove words containing digits.

    Args:
        text (str): The input text string to be cleaned.

    Returns:
        str: The cleaned text string with special characters removed.
    """
    try:
        text = re.sub(r"[^a-zA-Z0-9\s]", "", text)
        text = re.sub(r'http\S+|https?://\S+|www\.\S+', '', text)
        text = re.sub(r'\[.*?\]', '', text)
        text = re.sub(r'<.*?>+', '', text)
        text = re.sub(r'\n|\t|\r|\f|\b', '', text)
        text = re.sub(r'\w*\d\w*', '', text)
        return text.strip()
    except Exception as e:
        print(f"An error occurred while removing special characters: {e}")

def lowercase_text(text):
    """
    Convert text to lowercase.

    Args:
        text (str): The input text string to be converted.

    Returns:
        str: The text string converted to lowercase.
    """
    try:
        return text.lower()
    except Exception as e:
        print(f"An error occurred while converting text to lowercase: {e}")

def tokenize_text(text):
    """
    Tokenize text into words.

    Args:
        text (str): The input text string to be tokenized.

    Returns:
        list: A list of words tokenized from the input text.
    """
    try:
        return word_tokenize(text)
    except Exception as e:
        print(f"An error occurred while tokenizing text: {e}")

def remove_stopwords(text):
    """
    Remove stopwords from text.

    Args:
        text (str): The input text string from which stopwords will be removed.

    Returns:
        str: The text string with stopwords removed.
    """
    try:
        stop_words = set(stopwords.words('english'))
        words = text.split()
        filtered_words = [word for word in words if word.lower() not in stop_words]
        return ' '.join(filtered_words)
    except Exception as e:
        print(f"An error occurred while removing stopwords: {e}")

def stem_words(words):
    """
    Apply stemming to a list of words.

    Stemming reduces words to their base form using rule-based heuristics.

    Args:
        words (list): A list of words to be stemmed.

    Returns:
        list: A list of stemmed words.
    """
    try:
        from nltk.stem import PorterStemmer
        stemmer = PorterStemmer()
        return [stemmer.stem(word) for word in words]
    except Exception as e:
        print(f"An error occurred while stemming words: {e}")

def lemmatize_words(words):
    """
    Apply lemmatization to a list of words.

    Lemmatization reduces words to their dictionary form using linguistic rules.

    Args:
        words (list): A list of words to be lemmatized.

    Returns:
        list: A list of lemmatized words.
    """
    try:
        from nltk.stem import WordNetLemmatizer
        lemmatizer = WordNetLemmatizer()
        return [lemmatizer.lemmatize(word) for word in words]
    except Exception as e:
        print(f"An error occurred while lemmatizing words: {e}")

def remove_html_tags(text):
    """ 
    Remove HTML tags from text.

    Args:
        text (str): The input text string from which HTML tags will be removed.

    Returns:
        str: The text string with HTML tags removed.
    """
    try:
        clean_text = re.sub(r'<.*?>', '', text)
        return clean_text
    except Exception as e:
        print(f"An error occurred while removing HTML tags: {e}")

def remove_mention(text):
    """ 
    Remove @mentions from text.

    Args:
        text (str): The input text string from which @mentions will be removed.

    Returns:
        str: The text string with @mentions removed.
    """
    try:
        clean_text = re.sub(r'@\w+', '', text)
        return clean_text
    except Exception as e:
        print(f"An error occurred while removing mentions: {e}")

def remove_urls(text):
    """ 
    Remove URLs from text.

    Args:
        text (str): The input text string from which URLs will be removed.

    Returns:
        str: The text string with URLs removed.
    """
    try:
        clean_text = re.sub(r'http\S+', '', text)
        return clean_text
    except Exception as e:
        print(f"An error occurred while removing URLs: {e}")

def replace_chat_words(text):
    """
    Replace common chat words with their expanded forms.

    Args:
        text (str): The input text string containing chat words.

    Returns:
        str: The text string with chat words replaced by their expanded forms.
    """
    try:
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
    except Exception as e:
        print(f"An error occurred while replacing chat words: {e}")

def remove_punctuation(text):
    """
    Remove punctuation from text.

    Args:
        text (str): The input text string from which punctuation will be removed.

    Returns:
        str: The text string with punctuation removed.
    """
    try:
        clean_text = ''.join(ch for ch in text if ch not in string.punctuation)
        return clean_text
    except Exception as e:
        print(f"An error occurred while removing punctuation: {e}")

def remove_whitespace(text):
    """
    Remove leading and trailing whitespace from text.

    Args:
        text (str): The input text string from which whitespace will be removed.

    Returns:
        str: The text string with leading and trailing whitespace removed.
    """
    try:
        return text.strip()
    except Exception as e:
        print(f"An error occurred while removing whitespace: {e}")

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

    Args:
        text (str): The input text string to preprocess.
        remove_special (bool): Whether to remove special characters.
        to_lowercase (bool): Whether to convert the text to lowercase.
        remove_stopwords_flag (bool): Whether to remove stopwords from the text.
        stem_flag (bool): Whether to apply stemming.
        lemmatize_flag (bool): Whether to apply lemmatization.
        tokenize_flag (bool): Whether to tokenize the text into words.
        language (str): Language to use for stopwords and tokenization.

    Returns:
        str: Preprocessed text.
    """
    try:
        text = remove_html_tags(text)
        text = remove_urls(text)
        text = lowercase_text(text)
        text = replace_chat_words(text)
        text = remove_punctuation(text)
        text = remove_stopwords(text)
        text = remove_whitespace(text)
        text = remove_special_characters(text)
        return text
    except Exception as e:
        print(f"An error occurred during text preprocessing: {e}")

# Example usage:
# if __name__ == "__main__":
#     # Example text
#     text = """
#     This is an example sentence to demonstrate text preprocessing!
#     It includes punctuation, CAPITALIZED words, and stopwords.
#     """
#
#     processed_text = text_preprocessing(
#         text,
#         remove_special=True,
#         to_lowercase=True,
#         remove_stopwords_flag=False,
#         stem_flag=False,
#         lemmatize_flag=True,
#     )
#
#     print("Original Text:\n", text)
#     print("Processed Text:\n", processed_text)
