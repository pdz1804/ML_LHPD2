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
from sklearn.feature_selection import VarianceThreshold, SelectKBest, chi2
from sklearn.decomposition import LatentDirichletAllocation

class FeatureBuilder:
    """
    Manages feature extraction and transformation pipelines for text data.

    This class supports various vectorization methods (TF-IDF, Count, Word2Vec,
    GloVe, BERT), optional feature selection (Variance Threshold, Chi-squared,
    Topic Modeling), and optional dimensionality reduction (PCA, LDA). It allows
    fitting on training data and transforming both training and new data.
    Fitted components like vectorizers and reducers can be saved and loaded.

    Attributes:
        method (str): The primary feature extraction method to use (e.g., 'tfidf', 'bert').
        save_dir (str): Directory path to save/load fitted components (vectorizers, reducers).
        feature_selection (str or None): The feature selection technique to apply
            (e.g., 'variance', 'chi2', 'topic_modeling') or None to disable.
        reduce_dim (str or None): The dimensionality reduction technique ('pca', 'lda')
            or None to disable.
        n_components (int): Target number of dimensions/components for feature
            selection (like chi2, topic_modeling) or dimensionality reduction.
        vectorizer (object or None): Scikit-learn compatible vectorizer instance if
            `method` is 'tfidf', 'count', or 'binary_count'.
        word2vec_model (object or None): Loaded Gensim Word2Vec model if `method` is 'word2vec'.
        glove_model (object or None): Loaded Gensim GloVe model if `method` is 'glove'.
        tokenizer (object or None): Hugging Face tokenizer instance if `method` is 'bert'.
        bert_model (object or None): Hugging Face BERT model instance if `method` is 'bert'.
        reducer (object or None): Scikit-learn compatible dimensionality reducer instance
            (PCA or LDA) if `reduce_dim` is specified.
    """

    def __init__(self, method="tfidf", save_dir="data/processed", feature_selection=None, reduce_dim=None, n_components=100):
        """
        Initializes the FeatureBuilder with specified configurations.

        Sets up the chosen vectorization, feature selection, and dimensionality
        reduction methods based on the provided parameters. Loads pre-trained
        models (Word2Vec, GloVe, BERT) if required by the chosen method.

        Args:
            method (str, optional): The feature engineering method. Supported values:
                'tfidf', 'count', 'binary_count', 'word2vec', 'glove', 'bert'.
                Defaults to "tfidf".
            save_dir (str, optional): Directory path to save or load fitted components.
                Defaults to "data/processed".
            feature_selection (str or None, optional): Feature selection method.
                Supported: 'variance', 'chi2', 'topic_modeling', None. Defaults to None.
            reduce_dim (str or None, optional): Dimensionality reduction method.
                Supported: 'pca', 'lda', None. Defaults to None.
            n_components (int, optional): Target number of features/components for
                feature selection or dimensionality reduction. Defaults to 100.
        """
        self.method = method
        self.save_dir = save_dir
        self.feature_selection = feature_selection  # e.g., "variance", "chi2", "topic_modeling", or None
        self.reduce_dim = reduce_dim
        self.n_components = n_components
        self.reducer = None
        
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
            # self.glove_model = api.load("glove-wiki-gigaword-100")      # Pretrained GloVe embeddings
            self.glove_model = api.load("glove-wiki-gigaword-300")      # Pretrained GloVe embeddings
        elif method == "bert":
            self.tokenizer = AutoTokenizer.from_pretrained("sentence-transformers/all-MiniLM-L6-v2")
            self.bert_model = AutoModel.from_pretrained("sentence-transformers/all-MiniLM-L6-v2")
    
        # Initialize dimensionality reduction if required
        if self.reduce_dim == "pca":
            self.reducer = PCA(n_components=self.n_components)
        elif self.reduce_dim == "lda":
            self.reducer = LDA(n_components=self.n_components)

    def _apply_feature_selection(self, features, labels=None):
        """
        Applies the configured feature selection method to the feature matrix.

        Uses VarianceThreshold, SelectKBest (chi2), or LatentDirichletAllocation
        based on the `self.feature_selection` attribute.

        Args:
            features (np.ndarray or sparse matrix): The input feature matrix.
            labels (array-like, optional): The target labels, required only if
                `self.feature_selection` is 'chi2'. Defaults to None.

        Returns:
            np.ndarray or sparse matrix: The feature matrix after selection. Returns
                the original matrix if `self.feature_selection` is None.

        Raises:
            AssertionError: If 'chi2' selection is chosen but `labels` are not provided.
        """
        if self.feature_selection == "variance":
            selector = VarianceThreshold(threshold=0.01)
            return selector.fit_transform(features)
        elif self.feature_selection == "chi2":
            assert labels is not None, "Chi-squared feature selection requires class labels"
            selector = SelectKBest(chi2, k=self.n_components)
            return selector.fit_transform(features, labels)
        elif self.feature_selection == "topic_modeling":
            lda = LatentDirichletAllocation(n_components=self.n_components, random_state=42)
            return lda.fit_transform(features)
        else:
            return features

    def _apply_reducer(self, features, labels=None):
        """
        Applies the configured dimensionality reduction method to the feature matrix.

        Uses PCA or LDA based on the `self.reducer` attribute, fitting it first.

        Args:
            features (np.ndarray or sparse matrix): The input feature matrix.
            labels (array-like, optional): The target labels, required only if
                `self.reducer` is LDA. Defaults to None.

        Returns:
            np.ndarray: The dimensionally reduced feature matrix. Returns the
                original matrix if `self.reducer` is None.

        Raises:
            AssertionError: If LDA reduction is chosen but `labels` are not provided.
        """
        if self.reducer is not None:
            if isinstance(self.reducer, LDA):
                assert labels is not None, "LDA requires class labels during transform."
                features = self.reducer.fit_transform(features, labels)
            else:
                features = self.reducer.fit_transform(features)
        return features
    
    def _get_word2vec_vector(self, doc):
        """
        Computes the average Word2Vec vector for a single document.

        Looks up each token in the pre-loaded Word2Vec model and averages the
        vectors of the tokens found. Returns a zero vector if no tokens are found
        in the model's vocabulary.

        Args:
            doc (str): The input document text.

        Returns:
            np.ndarray: A 1D NumPy array representing the average Word2Vec embedding
                for the document. The array size matches the Word2Vec model's vector size.
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
        Computes the average GloVe vector for a single document.

        Looks up each token in the pre-loaded GloVe model and averages the
        vectors of the tokens found. Returns a zero vector if no tokens are found
        in the model's vocabulary.

        Args:
            doc (str): The input document text.

        Returns:
            np.ndarray: A 1D NumPy array representing the average GloVe embedding
                for the document. The array size matches the GloVe model's vector size.
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
        Computes the BERT sentence embedding (pooler output) for a single document.

        Uses the pre-loaded BERT tokenizer and model to obtain the embedding
        corresponding to the [CLS] token's representation after passing through
        the model layers.

        Args:
            doc (str): The input document text.

        Returns:
            np.ndarray: A 1D NumPy array representing the BERT embedding for the document.
        """
        inputs = self.tokenizer(doc, return_tensors="pt", padding=True, truncation=True, max_length=512)
        with torch.no_grad():
            outputs = self.bert_model(**inputs)
        return outputs.pooler_output.squeeze(0).numpy()
    
    def fit(self, texts):
        """
        Fits the vectorizer component to the training text data.

        For 'tfidf', 'count', and 'binary_count' methods, this involves learning the
        vocabulary and IDF weights (for TF-IDF). For embedding-based methods
        ('word2vec', 'glove', 'bert'), this method currently does nothing as
        pre-trained models are used. Fitting of dimensionality reducers or
        feature selectors happens during the `transform` or `fit_transform` step.

        Args:
            texts (list[str]): A list of raw text documents from the training set.
        """
        if self.method in ["tfidf", "count", "binary_count"]:
            self.vectorizer.fit(texts)

        elif self.method in ["word2vec", "glove", "bert"]:
            pass
            
    def transform(self, texts, labels=None):
        """
        Transforms the input text data into feature vectors using the fitted components.

        Applies the chosen vectorization method (`self.method`), followed by optional
        feature selection (`self.feature_selection`), and finally optional
        dimensionality reduction (`self.reduce_dim`). Dimensionality reduction models
        (PCA, LDA) are fitted within this step if not already fitted.

        Args:
            texts (list[str]): A list of raw text documents to transform.
            labels (list or np.ndarray, optional): Class labels corresponding to the `texts`.
                Required if using 'chi2' feature selection or 'lda' dimensionality reduction.
                Defaults to None.

        Returns:
            np.ndarray: The final transformed feature matrix.

        Raises:
            AssertionError: If 'lda' reduction or 'chi2' selection is requested but
                `labels` are not provided.
        """
        if self.method in ["tfidf", "count", "binary_count"]:
            # Transform the new data using the fitted vectorizer
            features = self.vectorizer.transform(texts).toarray()
            # return self._apply_reducer(features, labels)

        elif self.method == "word2vec":
            # Use the pre-trained Word2Vec model to generate embeddings
            word2vec_embeddings = []
            for doc in tqdm(texts, desc="Processing Word2Vec", unit="document"):
                word2vec_embeddings.append(self._get_word2vec_vector(doc))
            features = np.array(word2vec_embeddings)
            # return features

        elif self.method == "glove":
            # Similar process for GloVe embeddings
            glove_embeddings = []
            for doc in tqdm(texts, desc="Processing GloVe", unit="document"):
                glove_embeddings.append(self._get_glove_vector(doc))
            features = np.array(glove_embeddings)
            # return features

        elif self.method == "bert":
            # Use the pre-trained BERT model to generate embeddings
            bert_embeddings = []
            for doc in tqdm(texts, desc="Processing BERT", unit="document"):
                bert_embeddings.append(self._get_bert_embedding(doc))
            features = np.array(bert_embeddings)
            # return features
            
        # Optional feature selection
        features = self._apply_feature_selection(features, labels)

        # Apply dimensionality reduction if applicable
        # return self._apply_reducer(features, labels)
        if self.reduce_dim == "lda":
            assert labels is not None, "LDA requires class labels (y)."
            # features = self.vectorizer.transform(texts).toarray()
            self.reducer.fit(features, labels)
            return self.reducer.transform(features)
        elif self.reduce_dim == "pca":
            # features = self.vectorizer.transform(texts).toarray()
            self.reducer.fit(features)
            return self.reducer.transform(features)
        else:
            return features

    def fit_transform(self, texts, labels=None):
        """
        Fits the necessary components and transforms the text data in one step.

        Calls `fit()` to learn parameters (like vocabulary) from the texts,
        then calls `transform()` to generate the feature matrix, applying
        feature selection and dimensionality reduction as configured.

        Args:
            texts (list[str]): A list of raw text documents (typically the training set).
            labels (list or np.ndarray, optional): Class labels corresponding to the `texts`.
                Required if using 'chi2' feature selection or 'lda' dimensionality reduction.
                Defaults to None.

        Returns:
            np.ndarray: The final transformed feature matrix for the input texts.
        """
        self.fit(texts)  # First fit the model (compute parameters)
        return self.transform(texts, labels if self.reduce_dim == "lda" else None)  # Then transform the data using the fitted model
    
    def _save_model(self):
        """
        Saves the fitted components (vectorizer, reducer) to disk using pickle.

        Saves the relevant objects (Vectorizer for TF-IDF/Count methods,
        PCA/LDA reducer if used) to the directory specified by `self.save_dir`.
        File names are based on the method and reduction type.
        Note: For Word2Vec, GloVe, and BERT, it saves the potentially large models
        loaded in __init__, which might not be efficient if only the fitted
        vectorizer/reducer is needed later. Consider saving only fitted components.
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
        Loads previously saved components (vectorizer, reducer) from disk.

        Loads the pickled objects (Vectorizer, Reducer) from the `self.save_dir`
        based on the configured `self.method` and `self.reduce_dim`.
        Populates `self.vectorizer` and `self.reducer` attributes.
        Note: Also attempts to load saved Word2Vec/GloVe/BERT models, matching the
        behavior of `_save_model`.

        Raises:
            FileNotFoundError: If the expected saved file for the configured method
                or reducer does not exist in `self.save_dir`.
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

def build_vector_for_text(df_sampled, feature_methods, project_root, reduce_dim=None, n_components=50, feature_selection=None):
    """
    Builds feature vectors for train and test sets using multiple feature methods.

    Splits the input DataFrame into training and testing sets (stratified).
    Then, for each method specified in `feature_methods`, it initializes a
    `FeatureBuilder`, fits it on the training text, and transforms both the
    training and testing text data. Applies configured feature selection and
    dimensionality reduction.

    Args:
        df_sampled (pd.DataFrame): DataFrame containing at least 'text_clean' (str)
            and 'target' (int/categorical) columns.
        feature_methods (list[str]): A list of feature extraction method names
            (e.g., 'tfidf', 'bert') to apply. Must match methods supported by `FeatureBuilder`.
        project_root (str): The root directory of the project, used to construct
            the save path for processed data within the `FeatureBuilder`.
        reduce_dim (str or None, optional): Dimensionality reduction method ('pca', 'lda', None)
            to apply within `FeatureBuilder`. Defaults to None.
        n_components (int, optional): Target number of dimensions for feature selection
            or reduction. Defaults to 50. Adjusted for LDA based on number of classes.
        feature_selection (str or None, optional): Feature selection method ('variance',
            'chi2', 'topic_modeling', None) to apply within `FeatureBuilder`. Defaults to None.

    Returns:
        tuple: A tuple containing:
            - dict[str, pd.DataFrame]: Dictionary mapping each feature method name
              to its corresponding training feature matrix (as a DataFrame).
            - dict[str, pd.DataFrame]: Dictionary mapping each feature method name
              to its corresponding testing feature matrix (as a DataFrame).
            - pd.Series: The target labels for the training set.
            - pd.Series: The target labels for the testing set.
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
            # reduce_dim_method = reduce_dim if method in ["tfidf", "count", "binary_count"] else None
            reduce_dim_method = reduce_dim 

            feature_builder = FeatureBuilder(
                method=method,
                save_dir=os.path.join(project_root, "data", "processed"),
                feature_selection=feature_selection,  # Added feature selection parameter
                reduce_dim=reduce_dim_method,  # Only apply reduction to vector-based methods
                n_components=n_components
            )

            # Step 2: Fit on training data ONLY
            feature_builder.fit(df_train["text_clean"].tolist())

            # Step 3: Transform train and test sets separately
            # Pass labels when needed for feature selection (chi2) or reduction (lda)
            X_train = feature_builder.transform(
                df_train["text_clean"].tolist(),
                y_train.tolist() if (reduce_dim == "lda" or feature_selection == "chi2") else None
            )
            X_test = feature_builder.transform(
                df_test["text_clean"].tolist(),
                y_test.tolist() if (reduce_dim == "lda" or feature_selection == "chi2") else None
            )

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


