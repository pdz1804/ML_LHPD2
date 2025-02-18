"""
build_features.py

Author: Nguyen Quang Phu
Date: 2025-02-03
Updated: 2025-02-10
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

class FeatureBuilder:
    def __init__(self, method="tfidf", save_dir="data/processed", reduce_dim=None, n_components=100):
        """
        Initializes the FeatureBuilder with a specified feature engineering method.
        
        :param method: str, feature engineering method ('tfidf', 'count', 'word2vec', 'bert', etc.)
        :param save_dir: str, directory to save processed features
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
            self.glove_model = api.load("glove-wiki-gigaword-100")  # Pretrained GloVe embeddings
        elif method == "bert":
            self.tokenizer = AutoTokenizer.from_pretrained("sentence-transformers/all-MiniLM-L6-v2")
            self.bert_model = AutoModel.from_pretrained("sentence-transformers/all-MiniLM-L6-v2")
    
        # Initialize dimensionality reduction
        if self.reduce_dim == "pca":
            self.reducer = PCA(n_components=self.n_components)
        elif self.reduce_dim == "lda":
            self.reducer = LDA(n_components=min(self.n_components, 1))  # LDA needs class labels, adjust accordingly
    
    def _get_word2vec_vector(self, doc):
        """
        Extracts the average Word2Vec embedding for a document.

        :param doc: str, the document text
        :return: np.array, the averaged Word2Vec embedding
        """
        tokens = doc.split()
        word_vectors = []
        for token in tokens:
            if token in self.word2vec_model:  # Access word directly
                word_vectors.append(self.word2vec_model[token])  # No need for '.wv'
        if word_vectors:
            return np.mean(word_vectors, axis=0)
        else:
            return np.zeros(self.word2vec_model.vector_size)

    def _get_glove_vector(self, doc):
        """
        Extracts the average GloVe embedding for a document.

        :param doc: str, the document text
        :return: np.array, the averaged GloVe embedding
        """
        tokens = doc.split()
        word_vectors = []
        for token in tokens:
            if token in self.glove_model:  # Same for GloVe
                word_vectors.append(self.glove_model[token])  # Use directly without '.wv'
        if word_vectors:
            return np.mean(word_vectors, axis=0)
        else:
            return np.zeros(self.glove_model.vector_size)

    def _get_bert_embedding(self, doc):
        """
        Extracts the BERT embedding for a document.

        :param doc: str, the document text
        :return: np.array, the BERT embedding
        """
        inputs = self.tokenizer(doc, return_tensors="pt", padding=True, truncation=True, max_length=512)
        with torch.no_grad():
            outputs = self.bert_model(**inputs)
        return outputs.pooler_output.squeeze(0).numpy()
    
    def fit(self, texts, labels=None):
        """
        Fits the model to the text data by computing necessary statistics (e.g., vocabulary, embeddings).

        :param texts: list, raw text data
        :return: None
        """
        if self.method in ["tfidf", "count", "binary_count"]:
            self.vectorizer.fit(texts)
        elif self.method in ["word2vec", "glove", "bert"]:
            pass

        # if self.reduce_dim == "lda" and labels is not None:
        #     features = self.vectorizer.transform(texts).toarray()
        #     self.reducer.fit(features, labels)
        # elif self.reduce_dim == "pca":
        #     features = self.vectorizer.transform(texts).toarray()
        #     self.reducer.fit(features)

    def transform(self, texts, labels=None):
        """
        Transforms new data based on the fitted model.

        :param texts: list, raw text data
        :return: transformed feature matrix
        """
        if self.method in ["tfidf", "count", "binary_count"]:
            # Transform the new data using the fitted vectorizer
            features = self.vectorizer.transform(texts).toarray()

        elif self.method == "word2vec":
            # Use the pre-trained Word2Vec model to generate embeddings
            word2vec_embeddings = []
            for doc in tqdm(texts, desc="Processing Word2Vec", unit="document"):
                word2vec_embeddings.append(self._get_word2vec_vector(doc))
            features = np.array(word2vec_embeddings)

        elif self.method == "glove":
            # Similar process for GloVe embeddings
            glove_embeddings = []
            for doc in tqdm(texts, desc="Processing GloVe", unit="document"):
                glove_embeddings.append(self._get_glove_vector(doc))
            features = np.array(glove_embeddings)

        elif self.method == "bert":
            # Use the pre-trained BERT model to generate embeddings
            bert_embeddings = []
            for doc in tqdm(texts, desc="Processing BERT", unit="document"):
                bert_embeddings.append(self._get_bert_embedding(doc))
            features = np.array(bert_embeddings)

        # Apply dimensionality reduction if enabled
        if self.reduce_dim and features is not None:
            if self.reduce_dim == "lda" and labels is not None:
                # features = self.vectorizer.transform(texts).toarray()
                self.reducer.fit(features, labels)
            elif self.reduce_dim == "pca":
                # features = self.vectorizer.transform(texts).toarray()
                self.reducer.fit(features)
            
            features = self.reducer.transform(features)

        return features

    def fit_transform(self, texts):
        """
        Fits and transforms the text data by first fitting the model and then transforming it.

        :param texts: list, raw text data
        :return: transformed feature matrix
        """
        self.fit(texts)  # First fit the model (compute parameters)
        return self.transform(texts)  # Then transform the data using the fitted model
    
    def _save_model(self):
        """Saves the fitted vectorizer/scaler for later use."""
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
                
        if self.reduce_dim:
            reducer_path = os.path.join(self.save_dir, f"{self.reduce_dim}_reducer.pkl")
            with open(reducer_path, "wb") as f:
                pickle.dump(self.reducer, f)
    
    def _load_model(self):
        """Loads the previously saved vectorizer/scaler."""
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
            with open(reducer_path, "rb") as f:
                self.reducer = pickle.load(f)

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
#             print(f"✅ {method} - Loaded feature shape: {np.array(loaded_features).shape}")
            
#             # Display saved model file paths and contents
#             if method in ["tfidf", "count", "binary_count"]:
#                 model_file = os.path.join(feature_builder.save_dir, f"{method}_vectorizer.pkl")
#                 print(f"✅ {method} - Saved vectorizer file: {model_file}")
                
#                 # Print some content from the vectorizer (e.g., vocabulary)
#                 with open(model_file, "rb") as f:
#                     vectorizer = pickle.load(f)
#                     print(f"Sample vocabulary for {method}: {dict(list(vectorizer.vocabulary_.items())[:10])}")  # First 10 items
            
#             elif method in ["word2vec", "glove", "bert"]:
#                 print(f"✅ {method} - Model embeddings have been generated.")
                
#             print("\n")
#         except Exception as e:
#             print(f"❌ Error with method {method}: {e}\n")

#     # Load the models and display data
#     for method in methods:
#         try:
#             print(f"▶ Loading {method} model...\n")
            
#             # Define file paths for the saved models
#             if method in ["tfidf", "count", "binary_count"]:
#                 vectorizer_file = os.path.join("data/processed", f"{method}_vectorizer.pkl")
#                 with open(vectorizer_file, "rb") as f:
#                     vectorizer = pickle.load(f)
#                 print(f"✅ {method} - Vocabulary: {dict(list(vectorizer.vocabulary_.items())[:10])}")  # First 10 items

#             elif method in ["word2vec", "glove", "bert"]:
#                 model_file = os.path.join("data/processed", f"{method}_model.pkl")
#                 with open(model_file, "rb") as f:
#                     model = pickle.load(f)
#                 print(f"✅ {method} - Model loaded successfully. Model data type: {type(model)}")
                
#                 if method == "word2vec":
#                     print(f"✅ {method} - Example word embedding for 'handsome': {model['handsome'][:10]}")  # Show first 10 values
#                 elif method == "glove":
#                     # Assuming GloVe is a dictionary of words and embeddings
#                     print(f"✅ {method} - Example word embedding for 'handsome': {model['handsome'][:10]}")  # Show first 10 values
#                 elif method == "bert":
#                     # Load the BERT tokenizer separately
#                     tokenizer_file = os.path.join("data/processed", "bert_tokenizer.pkl")
#                     with open(tokenizer_file, "rb") as f:
#                         tokenizer = pickle.load(f)
#                     print(f"✅ {method} - Tokenizer loaded successfully. Tokenizer data type: {type(tokenizer)}")
                    
#                     # Use the tokenizer to encode the text
#                     input_ids = tokenizer.encode("handsome", return_tensors="pt")
#                     embeddings = model(input_ids).last_hidden_state
#                     print(f"✅ {method} - Example embedding for 'handsome': {embeddings[0][0][:10]}")  # First 10 values

#         except Exception as e:
#             print(f"❌ Error with method {method}: {e}\n")



