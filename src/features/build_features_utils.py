"""
build_features.py

Author: Nguyen Quang Phu
Date: 2025-02-03
Last Modified: 2025-02-25

This module includes:
- A FeatureBuilder class for feature extraction and transformation using various methods.
- A function to build feature vectors for text data.
"""

import os
import pickle
import torch
import numpy as np
import pandas as pd
from tqdm import tqdm
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
import gensim.downloader as api
from transformers import AutoTokenizer, AutoModel
from sklearn.decomposition import PCA
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
from sklearn.model_selection import train_test_split

class FeatureBuilder:
    """
    A class for feature extraction and transformation using various methods.

    Attributes:
        method (str): The feature extraction method ('tfidf', 'count', 'word2vec', 'bert', etc.).
        save_dir (str): Directory to save processed features.
        reduce_dim (str): Dimensionality reduction method ('pca', 'lda', or None).
        n_components (int): Number of components for dimensionality reduction.
        vectorizer (object): Vectorizer object for 'tfidf', 'count', or 'binary_count' methods.
        word2vec_model (object): Pretrained Word2Vec model.
        glove_model (object): Pretrained GloVe model.
        tokenizer (object): Tokenizer for BERT model.
        bert_model (object): BERT model for embedding extraction.
        reducer (object): Dimensionality reduction object (PCA or LDA).
    """

    def __init__(self, method="tfidf", save_dir="data/processed", reduce_dim=None, n_components=100):
        """
        Initializes the FeatureBuilder with a specified feature engineering method.

        Args:
            method (str): Feature engineering method ('tfidf', 'count', 'word2vec', 'bert', etc.).
            save_dir (str): Directory to save processed features.
            reduce_dim (str): Dimensionality reduction method ('pca', 'lda', or None).
            n_components (int): Number of components for dimensionality reduction.
        """
        self.method = method
        self.save_dir = save_dir
        self.reduce_dim = reduce_dim
        self.n_components = n_components
        os.makedirs(save_dir, exist_ok=True)
        
        # Define models for vectorization
        if method == "tfidf":
            self.vectorizer = TfidfVectorizer(max_features=2000, stop_words="english")
        elif method == "count":
            self.vectorizer = CountVectorizer(max_features=2000)
        elif method == "binary_count":
            self.vectorizer = CountVectorizer(binary=True, max_features=2000)
        elif method == "word2vec":
            self.word2vec_model = api.load("word2vec-google-news-300")  # Pretrained Google News Word2Vec
        elif method == "glove":
            self.glove_model = api.load("glove-wiki-gigaword-100")      # Pretrained GloVe embeddings
        elif method == "bert":
            self.tokenizer = AutoTokenizer.from_pretrained("sentence-transformers/all-MiniLM-L6-v2")
            self.bert_model = AutoModel.from_pretrained("sentence-transformers/all-MiniLM-L6-v2")
    
         # Initialize dimensionality reduction if required
        self.reducer = None
        if self.reduce_dim == "pca":
            self.reducer = PCA(n_components=self.n_components)
        elif self.reduce_dim == "lda":
            self.reducer = LDA(n_components=self.n_components)

    def _apply_reducer(self, features, labels=None):
        """Applies dimensionality reduction if enabled."""
        if self.reducer is not None:
            if isinstance(self.reducer, LDA):
                assert labels is not None, "LDA requires class labels during transform."
                features = self.reducer.fit_transform(features, labels)
            else:
                features = self.reducer.fit_transform(features)
        return features
    
    def _get_word2vec_vector(self, doc):
        """
        Extracts the average Word2Vec embedding for a document.

        Args:
            doc (str): The document text.

        Returns:
            np.array: The averaged Word2Vec embedding.
        """
        tokens = doc.split()
        word_vectors = []
        for token in tokens:
            if token in self.word2vec_model: 
                word_vectors.append(self.word2vec_model[token])  # No need for '.wv'
        if word_vectors:
            return np.mean(word_vectors, axis=0)
        else:
            return np.zeros(self.word2vec_model.vector_size)

    def _get_glove_vector(self, doc):
        """
        Extracts the average GloVe embedding for a document.

        Args:
            doc (str): The document text.

        Returns:
            np.array: The averaged GloVe embedding.
        """
        tokens = doc.split()
        word_vectors = []
        for token in tokens:
            if token in self.glove_model:  
                word_vectors.append(self.glove_model[token])  # Use directly without '.wv'
        if word_vectors:
            return np.mean(word_vectors, axis=0)
        else:
            return np.zeros(self.glove_model.vector_size)

    def _get_bert_embedding(self, doc):
        """
        Extracts the BERT embedding for a document.

        Args:
            doc (str): The document text.

        Returns:
            np.array: The BERT embedding.
        """
        inputs = self.tokenizer(doc, return_tensors="pt", padding=True, truncation=True, max_length=512)
        with torch.no_grad():
            outputs = self.bert_model(**inputs)
        return outputs.pooler_output.squeeze(0).numpy()
    
    def fit(self, texts, labels=None):
        """
        Fits the model to the text data by computing necessary statistics (e.g., vocabulary, embeddings).

        Args:
            texts (list): Raw text data.
            labels (list, optional): Class labels for LDA. Defaults to None.
        """
        if self.method in ["tfidf", "count", "binary_count"]:
            self.vectorizer.fit(texts)
            if self.reduce_dim == "lda":
                assert labels is not None, "LDA requires class labels (y)."
                features = self.vectorizer.transform(texts).toarray()
                self.reducer.fit(features, labels)
            elif self.reduce_dim == "pca":
                features = self.vectorizer.transform(texts).toarray()
                self.reducer.fit(features)

        elif self.method in ["word2vec", "glove", "bert"]:
            if self.reduce_dim == "lda":
                raise ValueError(f"LDA is not supported for method {self.method}")
            
    def transform(self, texts, labels=None):
        """
        Transforms new data based on the fitted model.

        Args:
            texts (list): Raw text data.
            labels (list, optional): Class labels for LDA. Defaults to None.

        Returns:
            np.array: Transformed feature matrix.
        """
        if self.method in ["tfidf", "count", "binary_count"]:
            # Transform the new data using the fitted vectorizer
            features = self.vectorizer.transform(texts).toarray()
            return self._apply_reducer(features, labels)

        elif self.method == "word2vec":
            # Use the pre-trained Word2Vec model to generate embeddings
            word2vec_embeddings = []
            for doc in tqdm(texts, desc="Processing Word2Vec", unit="document"):
                word2vec_embeddings.append(self._get_word2vec_vector(doc))
            features = np.array(word2vec_embeddings)
            return features

        elif self.method == "glove":
            # Similar process for GloVe embeddings
            glove_embeddings = []
            for doc in tqdm(texts, desc="Processing GloVe", unit="document"):
                glove_embeddings.append(self._get_glove_vector(doc))
            features = np.array(glove_embeddings)
            return features

        elif self.method == "bert":
            # Use the pre-trained BERT model to generate embeddings
            bert_embeddings = []
            for doc in tqdm(texts, desc="Processing BERT", unit="document"):
                bert_embeddings.append(self._get_bert_embedding(doc))
            features = np.array(bert_embeddings)
            return features

        # Apply dimensionality reduction if applicable
        # return self._apply_reducer(features, labels)

    def fit_transform(self, texts):
        """
        Fits and transforms the text data by first fitting the model and then transforming it.

        Args:
            texts (list): Raw text data.

        Returns:
            np.array: Transformed feature matrix.
        """
        self.fit(texts)  # First fit the model (compute parameters)
        return self.transform(texts)  # Then transform the data using the fitted model
    
    def _save_model(self):
        """
        Saves the fitted vectorizer/scaler for later use.
        """
        # Ensure the directory exists
        save_dir = self.save_dir if self.save_dir else "data/processed"
        os.makedirs(save_dir, exist_ok=True)  # Create directory if it doesn't exist  
        
        if self.method in ["tfidf", "count", "binary_count"]:
            file_path = os.path.join(self.save_dir, f"{self.method}_vectorizer.pkl")
            with open(file_path, "wb") as f:
                pickle.dump(self.vectorizer, f)
        elif self.method in ["word2vec", "glove"]:
            # Save the Word2Vec or GloVe model
            file_path = os.path.join(self.save_dir, f"{self.method}_model.pkl")
            with open(file_path, "wb") as f:
                if self.method == "word2vec":
                    pickle.dump(self.word2vec_model, f)
                elif self.method == "glove":
                    pickle.dump(self.glove_model, f)
        elif self.method == "bert":
            # Save the BERT tokenizer and model
            tokenizer_path = os.path.join(self.save_dir, "bert_tokenizer.pkl")
            model_path = os.path.join(self.save_dir, "bert_model.pkl")
            with open(tokenizer_path, "wb") as f:
                pickle.dump(self.tokenizer, f)
            with open(model_path, "wb") as f:
                pickle.dump(self.bert_model, f)
                
        if self.reducer is not None:
            reducer_path = os.path.join(self.save_dir, f"{self.reduce_dim}_reducer.pkl")
            with open(reducer_path, "wb") as f:
                pickle.dump(self.reducer, f)
    
    def _load_model(self):
        """
        Loads the previously saved vectorizer/scaler.
        """
        # Ensure the directory exists
        os.makedirs(self.save_dir, exist_ok=True)
        
        if self.method in ["tfidf", "count", "binary_count"]:
            file_path = os.path.join(self.save_dir, f"{self.method}_vectorizer.pkl")
            if not os.path.exists(file_path):
                raise FileNotFoundError(f"No saved model found at {file_path}. Run `fit_transform` first.")
            with open(file_path, "rb") as f:
                self.vectorizer = pickle.load(f)
        elif self.method in ["word2vec", "glove"]:
            file_path = os.path.join(self.save_dir, f"{self.method}_model.pkl")
            if not os.path.exists(file_path):
                raise FileNotFoundError(f"No saved model found at {file_path}. Run `fit_transform` first.")
            with open(file_path, "rb") as f:
                self.word2vec_model = pickle.load(f)
        elif self.method == "bert":
            tokenizer_path = os.path.join(self.save_dir, "bert_tokenizer.pkl")
            model_path = os.path.join(self.save_dir, "bert_model.pkl")
            if not os.path.exists(tokenizer_path) or not os.path.exists(model_path):
                raise FileNotFoundError(f"No saved BERT model found at {tokenizer_path} or {model_path}. Run `fit_transform` first.")
            with open(tokenizer_path, "rb") as f:
                self.tokenizer = pickle.load(f)
            with open(model_path, "rb") as f:
                self.bert_model = pickle.load(f)
        
        if self.reduce_dim:
            reducer_path = os.path.join(self.save_dir, f"{self.reduce_dim}_reducer.pkl")
            if not os.path.exists(reducer_path):
                raise FileNotFoundError(f"No saved reducer found at {reducer_path}.")
            with open(reducer_path, "rb") as f:
                self.reducer = pickle.load(f)

def build_vector_for_text(df_sampled, feature_methods, project_root, reduce_dim=None, n_components=50):
    """
    Builds feature vectors for text data using specified feature extraction methods.

    Args:
        df_sampled (pd.DataFrame): The sampled DataFrame containing text data.
        feature_methods (list): List of feature extraction methods to use.
        project_root (str): Root directory of the project.

    Returns:
        dict: Dictionary of training feature matrices for each method.
        dict: Dictionary of testing feature matrices for each method.
        pd.Series: Training labels.
        pd.Series: Testing labels.
    """
    X_train_features_dict = {}
    X_test_features_dict = {}

    # Step 1: First, split the DataFrame before feature extraction (to maintain X-y matching)
    df_train, df_test = train_test_split(df_sampled, test_size=0.2, random_state=42, stratify=df_sampled["target"])

    # Extract y_train and y_test **before feature extraction** to ensure data alignment
    y_train = df_train["target"].reset_index(drop=True)
    y_test = df_test["target"].reset_index(drop=True)

    print("\n🔎 Running feature extraction...\n")
    for method in tqdm(feature_methods, desc="Feature Extraction Progress"):
        print(f"\n🔍 Processing feature extraction using: {method}...")

        try:
            n_classes = len(y_train.unique())
            if reduce_dim == "lda":
                n_components = min(n_components, n_classes - 1)
                
            # Initialize FeatureBuilder for the current method
            reduce_dim_method = reduce_dim if method in ["tfidf", "count", "binary_count"] else None

            feature_builder = FeatureBuilder(
                method=method,
                save_dir=os.path.join(project_root, "data", "processed"),
                reduce_dim=reduce_dim_method,  # Only apply reduction to vector-based methods
                n_components=n_components
            )

            # Step 2: Fit on training data ONLY
            feature_builder.fit(df_train["text_clean"].tolist(), y_train if reduce_dim == "lda" else None)

            # Step 3: Transform train and test sets separately
            X_train = feature_builder.transform(df_train["text_clean"].tolist(), n_classes if reduce_dim == "lda" else None)
            X_test = feature_builder.transform(df_test["text_clean"].tolist())

            # Ensure feature matrices are DataFrames
            X_train_features_dict[method] = pd.DataFrame(X_train)
            X_test_features_dict[method] = pd.DataFrame(X_test)

            print(f"✅ {method} - Train shape: {X_train.shape}, Test shape: {X_test.shape}")

        except Exception as e:
            print(f"❌ Error with {method}: {e}. Skipping this method.")

    return X_train_features_dict, X_test_features_dict, y_train, y_test

# if __name__ == "__main__":
#     # Sample texts for testing
#     sample_texts = [
#         "The quick brown fox jumps over the lazy dog.",
#         "I love machine learning and natural language processing!",
#         "Deep learning models are revolutionizing AI applications."
#     ]
    
#     # List of feature engineering methods to test
#     methods = ["tfidf", "count", "binary_count", "word2vec", "glove", "bert"]

#     print("\n🔍 Running tests on feature extraction methods...\n")
    
#     for method in methods:
#         try:
#             print(f"▶ Testing method: {method}...")

#             # Reload the model and transform data again
#             feature_builder = FeatureBuilder(method=method, save_dir="data/processed")
#             loaded_features = feature_builder.fit_transform(sample_texts)
#             feature_builder._save_model()  # Save the model for later use
#             print(f"{method} - Loaded feature shape: {np.array(loaded_features).shape}")
            
#             # Display saved model file paths and contents
#             if method in ["tfidf", "count", "binary_count"]:
#                 model_file = os.path.join(feature_builder.save_dir, f"{method}_vectorizer.pkl")
#                 print(f"{method} - Saved vectorizer file: {model_file}")
                
#                 # Print some content from the vectorizer (e.g., vocabulary)
#                 with open(model_file, "rb") as f:
#                     vectorizer = pickle.load(f)
#                     print(f"Sample vocabulary for {method}: {dict(list(vectorizer.vocabulary_.items())[:10])}")  # First 10 items
            
#             elif method in ["word2vec", "glove", "bert"]:
#                 print(f"{method} - Model embeddings have been generated.")
                
#             print("\n")
#         except Exception as e:
#             print(f"Error with method {method}: {e}\n")


