"""
build_features.py

Author: Nguyen Quang Phu
Date: 2025-02-03
Updated: 2025-02-10

This module includes:
- Functions for creating and training various machine learning models.
- Functions for feature selection using genetic algorithms.
- Functions for training and evaluating deep learning models (CNN, LSTM).
- Functions for training graphical models (HMM, Bayesian Network).
"""

import os
import joblib
import pandas as pd
import numpy as np
from tqdm import tqdm
from statistics import mean
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, roc_auc_score
from sklearn.preprocessing import StandardScaler, MaxAbsScaler, MinMaxScaler, KBinsDiscretizer
from sklearn.model_selection import KFold, GridSearchCV, cross_val_score
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.linear_model import Perceptron, LogisticRegression
from sklearn.svm import SVC
from sklearn.naive_bayes import GaussianNB
from sklearn.decomposition import PCA
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
import hmmlearn.hmm
import xgboost as xgb
from sklearn.ensemble import RandomForestClassifier, VotingClassifier, StackingClassifier
from sklearn_crfsuite import CRF
from sklearn.metrics import log_loss, hinge_loss
from sklearn.metrics import classification_report

from sklearn.base import BaseEstimator, ClassifierMixin

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import keras_tuner as kt

from src.features.build_features_utils import *

import keras
from keras import layers
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from sklearn.model_selection import train_test_split
from keras_tuner import RandomSearch

from pgmpy.models import BayesianNetwork
from pgmpy.estimators import MaximumLikelihoodEstimator
from pgmpy.inference import VariableElimination, BeliefPropagation

import nltk
import re
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from collections import defaultdict, Counter
from hmmlearn import hmm

from pgmpy.readwrite import BIFReader

nltk.download('punkt')
nltk.download('stopwords')

import keras_nlp
from tensorflow.keras.optimizers import Adam

from statistics import mean

# added
from transformers import BertTokenizer, TFBertForSequenceClassification

# --------------------------------------------------
# Loc defined
class BayesianNetworkClassifier(BaseEstimator, ClassifierMixin):
    """
    BayesianNetworkClassifier

    This module provides a custom Bayesian Network classifier implementing scikit-learn's BaseEstimator and ClassifierMixin.
    The classifier supports feature selection, discretization, dimensionality reduction, and inference using Bayesian Networks.

    Key functionalities:
    - Feature filtering based on unique values
    - Principal Component Analysis (PCA) for dimensionality reduction
    - Discretization of continuous features
    - Training a Bayesian Network using Maximum Likelihood Estimation
    - Inference using Belief Propagation or Variable Elimination
    - Compatibility with scikit-learn's API for easy integration
    """
    
    def __init__(self, structure=None, n_bins=2, strategy='kmeans', min_unique_values=2, max_features=20):
        """
        Initializes the BayesianNetworkClassifier with user-defined parameters.

        Args:
            structure (list, optional): A predefined structure for the Bayesian Network.
            n_bins (int, optional): Number of bins for discretization (default is 2).
            strategy (str, optional): Discretization strategy (default is 'kmeans').
            min_unique_values (int, optional): Minimum unique values required per feature (default is 2).
            max_features (int, optional): Maximum features allowed after PCA (default is 20).
        """
        self.structure = structure
        self.n_bins = n_bins
        self.strategy = strategy
        self.min_unique_values = min_unique_values
        self.max_features = max_features
        self.model = None
        self.inference = None
        self.feature_names = None
        self.discretizer = None
        self.pca = None
        self.filtered_columns = None  # Lưu cột sau khi lọc
    
    def fit(self, X, y):
        """
        Fits the Bayesian Network classifier to the training data.

        Steps:
        1. Converts `X` into a DataFrame if necessary.
        2. Filters out low-variance features with fewer than `min_unique_values` unique values.
        3. Applies PCA if the number of features exceeds `max_features`.
        4. Discretizes the features based on the selected strategy.
        5. Trains the Bayesian Network using Maximum Likelihood Estimation.
        6. Initializes the inference engine for predictions.

        Args:
            X (array-like or pd.DataFrame): Feature matrix.
            y (array-like): Target labels.

        Returns:
            self: Trained model instance.
        """
        if isinstance(X, np.ndarray):
            X = pd.DataFrame(X, columns=[f"col_{i}" for i in range(X.shape[1])])
        
        # Loại bỏ các đặc trưng không đủ đa dạng
        unique_counts = X.nunique()
        self.filtered_columns = unique_counts[unique_counts >= self.min_unique_values].index.tolist()
        if len(self.filtered_columns) < 1:
            raise ValueError(f"No features with at least {self.min_unique_values} unique values.")
        X = X[self.filtered_columns]
        
        print(f"After filtering: {len(self.filtered_columns)} features remain.")
        print("Unique values per feature before discretization:\n", X.nunique())
        
        # Giảm số đặc trưng bằng PCA
        if len(self.filtered_columns) > self.max_features:
            self.pca = PCA(n_components=self.max_features)
            X_reduced = self.pca.fit_transform(X)
            X = pd.DataFrame(X_reduced, columns=[f"feat_{i}" for i in range(self.max_features)])
            self.feature_names = X.columns.tolist()
            print(f"After PCA: Reduced to {self.max_features} features.")
        else:
            self.feature_names = self.filtered_columns
        
        # Rời rạc hóa dữ liệu
        is_discrete = all(X[col].apply(lambda x: x.is_integer() if pd.notna(x) else True).all() for col in X.columns)
        if is_discrete:
            X_discrete = X.astype(int)
        else:
            if X.max().max() < 1e-5:
                X_discrete = (X > 0).astype(int)
            else:
                self.discretizer = KBinsDiscretizer(n_bins=self.n_bins, encode='ordinal', strategy=self.strategy)
                X_discrete = pd.DataFrame(self.discretizer.fit_transform(X), columns=self.feature_names)
        
        print("Unique values per feature after discretization:\n", X_discrete.nunique())
        
        # Kiểm tra dữ liệu và nhãn
        unique_values_per_feature = X_discrete.nunique()
        if any(unique_values_per_feature < 2):
            raise ValueError("Some features have fewer than 2 unique values after processing.")
        y = pd.Series(y).astype(int).reset_index(drop=True)
        if y.nunique() < 2:
            raise ValueError(f"Label has fewer than 2 unique values in this fold: {y.value_counts().to_dict()}")
        
        # Tạo dữ liệu huấn luyện
        data = X_discrete.copy()
        data['label'] = y
        
        # Tạo cấu trúc đơn giản
        if self.structure is None:
            self.structure = [(feat, 'label') for feat in self.feature_names]
        
        # Huấn luyện mô hình
        self.model = BayesianNetwork(self.structure)
        self.model.fit(data, estimator=MaximumLikelihoodEstimator)
        # self.inference = VariableElimination(self.model)
        self.inference = BeliefPropagation(self.model)
        return self
    
    def predict(self, X):
        """
        Predicts class labels for the given test data.

        Steps:
        1. Converts `X` into a DataFrame if necessary.
        2. Applies the same filtering and transformations as in `fit()`.
        3. Performs inference using the Bayesian Network.
        
        Args:
            X (array-like or pd.DataFrame): Test feature matrix.

        Returns:
            np.array: Predicted class labels.
        """
        if isinstance(X, np.ndarray):
            X = pd.DataFrame(X, columns=[f"col_{i}" for i in range(X.shape[1])])
        
        # Lọc dữ liệu kiểm tra bằng các cột đã lọc trong fit()
        X = X[self.filtered_columns]
        
        # Áp dụng PCA nếu có
        if self.pca is not None:
            X = pd.DataFrame(self.pca.transform(X), columns=self.feature_names)
        
        # Áp dụng rời rạc hóa nếu có
        if self.discretizer is not None:
            X_discrete = pd.DataFrame(self.discretizer.transform(X), columns=self.feature_names)
        else:
            X_discrete = X.astype(int)
        
        # Dự đoán
        y_pred = []
        for i in range(len(X_discrete)):
            evidence = {k: v for k, v in X_discrete.iloc[i].to_dict().items() if pd.notna(v)}
            pred = self.inference.map_query(variables=['label'], evidence=evidence, show_progress=False)
            y_pred.append(pred['label'])
        return np.array(y_pred)
    
    def score(self, X, y):
        """
        Computes the accuracy of the classifier.

        Args:
            X (array-like or pd.DataFrame): Test feature matrix.
            y (array-like): True labels.

        Returns:
            float: Accuracy score.
        """
        y_pred = self.predict(X)
        return accuracy_score(y, y_pred)
    
    def get_params(self, deep=True):
        """
        Returns model parameters in a dictionary format.

        Args:
            deep (bool, optional): Whether to return parameters for sub-objects (default is True).

        Returns:
            dict: Model parameters.
        """
        return {"structure": self.structure, "n_bins": self.n_bins, "strategy": self.strategy, 
                "min_unique_values": self.min_unique_values, "max_features": self.max_features}
    
    def set_params(self, **params):
        """
        Sets model parameters dynamically.

        Args:
            **params: Keyword arguments containing parameter names and values.

        Returns:
            self: Model instance with updated parameters.
        """
        for param, value in params.items():
            setattr(self, param, value)
        return self

# --------------------------------------------------
# Hung defined
def create_population(num_features, population_size):
    """
    Creates an initial population of binary feature selectors.

    Args:
        num_features (int): Number of features.
        population_size (int): Size of the population.

    Returns:
        np.ndarray: Initial population of binary feature selectors.
    """
    return np.random.randint(2, size=(population_size, num_features))

def fitness_function(features, X_train, y_train):
    """
    Evaluates the fitness of a feature selection candidate.

    Args:
        features (np.ndarray): Binary feature selector.
        X_train (np.ndarray): Training feature matrix.
        y_train (np.ndarray): Training labels.

    Returns:
        float: Fitness score (cross-validation accuracy).
    """
    selected_features = [i for i, f in enumerate(features) if f == 1]
    if not selected_features:  # Avoid empty feature sets
        return 0

    X_train_selected = X_train[:, selected_features]

    nb_model = GaussianNB(var_smoothing=1e-8)
    try:
        scores = cross_val_score(nb_model, X_train_selected, y_train, cv=5)
        return np.mean(scores)
    except ValueError as e:
        print(f"Error during cross-validation: {e}")
        return 0

def crossover(parent1, parent2):
    """
    Performs single-point crossover.

    Args:
        parent1 (np.ndarray): First parent.
        parent2 (np.ndarray): Second parent.

    Returns:
        tuple: Two offspring resulting from the crossover.
    """
    point = np.random.randint(1, len(parent1) - 1)
    offspring1 = np.concatenate((parent1[:point], parent2[point:]))
    offspring2 = np.concatenate((parent2[:point], parent1[point:]))
    return offspring1, offspring2

def mutate(individual, mutation_rate=0.1):
    """
    Mutates an individual with a given probability.

    Args:
        individual (np.ndarray): Individual to mutate.
        mutation_rate (float): Probability of mutation.

    Returns:
        np.ndarray: Mutated individual.
    """
    for i in range(len(individual)):
        if np.random.rand() < mutation_rate:
            individual[i] = 1 - individual[i]
    return individual

def genetic_algorithm(X_train, y_train, X_test, y_test, model_save_path=None, img_save_path=None, img_loss_path=None, population_size=20, num_generations=100, mutation_rate=0.1, crossover_rate=0.7):
    """
    Runs a genetic algorithm to optimize feature selection for Naive Bayes.

    Args:
        X_train (pd.DataFrame): Training feature matrix.
        y_train (pd.Series): Training labels.
        X_test (pd.DataFrame): Testing feature matrix.
        y_test (pd.Series): Testing labels.
        model_save_path (str, optional): Path to save the trained model. Defaults to None.
        population_size (int): Size of the population. Defaults to 20.
        num_generations (int): Number of generations. Defaults to 100.
        mutation_rate (float): Probability of mutation. Defaults to 0.1.
        crossover_rate (float): Probability of crossover. Defaults to 0.7.

    Returns:
        GaussianNB: Trained Naive Bayes model.
    """
    # Check if the model already exists
    if os.path.exists(model_save_path):
        print(f"🔄 Loading existing model from {model_save_path}...")
        model_algorithm = joblib.load(model_save_path)
        return model_algorithm
    
    num_features = X_train.shape[1]
    population = create_population(num_features, population_size)

    for generation in range(num_generations):
        fitness_scores = [fitness_function(ind, X_train.values, y_train) for ind in population]
        
        # Normalize fitness scores to avoid division errors
        fitness_scores = np.array(fitness_scores)
        fitness_scores = np.clip(fitness_scores, 1e-5, None)
        
        probabilities = fitness_scores / np.sum(fitness_scores)

        # Select parents based on probabilities
        selected_indices = np.random.choice(np.arange(population_size), size=population_size, p=probabilities)
        selected_parents = [population[idx] for idx in selected_indices]

        next_generation = []
        for j in range(0, population_size, 2):
            if np.random.rand() < crossover_rate:
                offspring1, offspring2 = crossover(selected_parents[j], selected_parents[j + 1])
            else:
                offspring1, offspring2 = selected_parents[j], selected_parents[j + 1]
            
            next_generation.append(mutate(offspring1, mutation_rate))
            next_generation.append(mutate(offspring2, mutation_rate))

        population = next_generation  # Move to the next generation

    # Select the best individual
    best_individual = population[np.argmax(fitness_scores)]
    selected_features = [i for i, f in enumerate(best_individual) if f == 1]

    X_train_selected = X_train.iloc[:, selected_features]
    X_test_selected = X_test.iloc[:, selected_features]

    print(f"Selected {len(selected_features)} features out of {num_features}")

    # Scaling the selected features
    scaler = MinMaxScaler()
    X_train_selected = scaler.fit_transform(X_train_selected)
    X_test_selected = scaler.transform(X_test_selected)

    # Train Naive Bayes with selected features
    nb_model = GaussianNB()
    
    # Set up K-Fold Cross-Validation
    print("\n🎯 Running K-Fold Cross-Validation...")
    k_fold = KFold(n_splits=5, shuffle=True, random_state=42)

    # Prepare containers for metrics
    accuracy_scores = []
    roc_auc_scores = []
    f1_scores = []
    precision_scores = []
    recall_scores = []
    training_losses = []
    validation_losses = []

    # Perform CV loop
    for train_index, val_index in tqdm(k_fold.split(X_train_selected), total=k_fold.get_n_splits(), desc="K-Fold Progress"):
        X_train_cv, X_val_cv = X_train_selected[train_index], X_train_selected[val_index]
        y_train_cv, y_val_cv = y_train[train_index], y_train[val_index]
        
        model_cv = GaussianNB()
        model_cv.fit(X_train_cv, y_train_cv)

        # Loss
        train_loss = get_training_loss(model_cv, X_train_cv, y_train_cv)
        val_loss = get_training_loss(model_cv, X_val_cv, y_val_cv)
        training_losses.append(train_loss)
        validation_losses.append(val_loss)

        # Predictions
        val_preds = model_cv.predict(X_val_cv)
        val_probs = model_cv.predict_proba(X_val_cv)[:, 1]

        # Metrics
        accuracy_scores.append(accuracy_score(y_val_cv, val_preds))
        roc_auc_scores.append(roc_auc_score(y_val_cv, val_probs))
        f1_scores.append(f1_score(y_val_cv, val_preds))
        precision_scores.append(precision_score(y_val_cv, val_preds))
        recall_scores.append(recall_score(y_val_cv, val_preds))

    # Print average scores
    print(f'📊 Average Accuracy: {int(mean(accuracy_scores) * 100)}%')
    print(f'📊 Average ROC AUC: {int(mean(roc_auc_scores) * 100)}%')
    print(f'📊 Average F1 Score: {int(mean(f1_scores) * 100)}%')
    print(f'📊 Average Precision: {int(mean(precision_scores) * 100)}%')
    print(f'📊 Average Recall: {int(mean(recall_scores) * 100)}%')

    # Retrain final model on full training set
    nb_model.fit(X_train_selected, y_train)

    # Save model
    if model_save_path:
        joblib.dump(nb_model, model_save_path)
        print(f'💾 Model saved to {model_save_path}')

    # Plot accuracy & ROC AUC per fold
    if img_save_path:
        plt.figure(figsize=(10, 6))
        plt.plot(range(1, len(accuracy_scores) + 1), accuracy_scores, label="Accuracy", marker='o')
        plt.plot(range(1, len(roc_auc_scores) + 1), roc_auc_scores, label="ROC AUC", marker='o')
        plt.title("Validation Performance Across K-Folds")
        plt.xlabel("Fold Number")
        plt.ylabel("Score")
        plt.legend()
        plt.savefig(img_save_path)
        plt.close()
        print(f"📈 Plot saved to {img_save_path}")

    # Plot loss curves
    if img_loss_path:
        plt.figure(figsize=(10, 6))
        plt.plot(range(1, len(training_losses) + 1), training_losses, label="Training Loss", marker='o')
        plt.plot(range(1, len(validation_losses) + 1), validation_losses, label="Validation Loss", marker='o')
        plt.title("Training & Validation Loss Across K-Folds")
        plt.xlabel("Fold Number")
        plt.ylabel("Loss")
        plt.legend()
        plt.savefig(img_loss_path)
        plt.close()
        print(f"📉 Loss plot saved to {img_loss_path}")

# --------------------------------------------------

def generate_binary_classification_model(X, y, model_algorithm, hyperparameters, needs_scaled = False, model_save_path="best_model.pkl", img_save_path=None, img_loss_path=None):
    """
    Generates everything required for training and validation of a binary classification model.

    Args:
        X (pd.DataFrame): Training features.
        y (pd.Series): Target values.
        model_algorithm (object): Model algorithm to train.
        hyperparameters (dict): Hyperparameters for tuning.
        needs_scaled (bool): Whether to scale the dataset. Defaults to False.
        model_save_path (str): Path to save the best model. Defaults to "best_model.pkl".
        img_save_path (str, optional): Path to save validation performance plot. Defaults to None.
        img_loss_path (str, optional): Path to save training loss plot. Defaults to None.

    Returns:
        object: Trained model.
    """
    # Check if the model already exists
    if os.path.exists(model_save_path):
        print(f"🔄 Loading existing model from {model_save_path}...")
        model_algorithm = joblib.load(model_save_path)
        return model_algorithm
    
    print(f"🚀 Training new model: {model_algorithm.__class__.__name__}...")
    # Performing a scaling on the data if required
    if needs_scaled:
        
        # Instantiating the StandardScaler
        scaler = StandardScaler()
        
        # Performing a fit_transform on the dataset
        scaled_features = scaler.fit_transform(X)
        
        # Transforming the StandardScaler output back into a Pandas DataFrame
        X = pd.DataFrame(scaled_features, index = X.index, columns = X.columns)
        
    # Instantiating a GridSearch object with the inputted model algorithm and hyperparameters
    gridsearchcv = GridSearchCV(estimator = model_algorithm,
                                param_grid = hyperparameters)
    
    # Fitting the training data to the GridSearch object
    gridsearchcv.fit(X, y)
    
    # Printing out the best hyperparameters
    print(f'Best hyperparameters: {gridsearchcv.best_params_}')
    
    # Instantiating a new model object with the ideal hyperparameters from the GridSearch job
    model_algorithm.set_params(**gridsearchcv.best_params_)
    
    # Creating a container to hold each set of validation metrics
    accuracy_scores, roc_auc_scores, f1_scores, precision_scores, recall_scores = [], [], [], [], []
    training_losses, validation_losses = [], []
    
    # Instantiating the K-Fold cross validation object
    k_fold = KFold(n_splits = 5)
    
    print("\n🎯 Running K-Fold Cross-Validation...")
    for train_index, val_index in tqdm(k_fold.split(X), total=k_fold.get_n_splits(), desc="K-Fold Progress"):

        # Splitting the training set from the validation set for this specific fold
        X_train, X_val = X.iloc[train_index, :], X.iloc[val_index, :]
        y_train, y_val = y.iloc[train_index], y.iloc[val_index]
        
        # Fitting the X_train and y_train datasets to the model algorithm
        model_algorithm.fit(X_train, y_train)
        
        # Compute losses
        train_loss = get_training_loss(model_algorithm, X_train, y_train)
        val_loss = get_training_loss(model_algorithm, X_val, y_val)

        training_losses.append(train_loss)
        validation_losses.append(val_loss)

        # Getting inferential predictions for the validation dataset
        val_preds = model_algorithm.predict(X_val)

        # Generating validation metrics by comparing the inferential predictions (val_preds) to the actuals (y_val)
        val_accuracy = accuracy_score(y_val, val_preds)
        val_roc_auc_score = roc_auc_score(y_val, val_preds)
        val_f1_score = f1_score(y_val, val_preds)
        val_precision_score = precision_score(y_val, val_preds)
        val_recall_score = recall_score(y_val, val_preds)
        
        # Appending the validation scores to the respective validation metric container
        accuracy_scores.append(val_accuracy)
        roc_auc_scores.append(val_roc_auc_score)
        f1_scores.append(val_f1_score)
        precision_scores.append(val_precision_score)
        recall_scores.append(val_recall_score)
        
    # Print average validation scores
    print(f'📊 Average Accuracy: {int(mean(accuracy_scores) * 100)}%')
    print(f'📊 Average ROC AUC: {int(mean(roc_auc_scores) * 100)}%')
    print(f'📊 Average F1 Score: {int(mean(f1_scores) * 100)}%')
    print(f'📊 Average Precision: {int(mean(precision_scores) * 100)}%')
    print(f'📊 Average Recall: {int(mean(recall_scores) * 100)}%')
    
    # New added
    model_algorithm.fit(X, y)

    # Save the trained model
    joblib.dump(model_algorithm, model_save_path)
    print(f'💾 Model saved to {model_save_path}')
    
    # If img_save_path is provided, plot the validation scores
    if img_save_path:
        plt.figure(figsize=(10, 6))

        # Plot Accuracy
        plt.plot(range(1, len(accuracy_scores) + 1), accuracy_scores, label="Accuracy", marker='o')

        # Plot ROC AUC
        plt.plot(range(1, len(roc_auc_scores) + 1), roc_auc_scores, label="ROC AUC", marker='o')

        # Add labels and title
        plt.title("Validation Performance Across K-Folds")
        plt.xlabel("Fold Number")
        plt.ylabel("Score")
        plt.legend()

        # Save the plot to the specified path
        plt.savefig(img_save_path)
        plt.close()
        print(f"📈 Plot saved to {img_save_path}")
        
    # Plot loss curves
    if img_loss_path:
        plt.figure(figsize=(10, 6))
        plt.plot(range(1, len(training_losses) + 1), training_losses, label="Training Loss", marker='o')
        plt.plot(range(1, len(validation_losses) + 1), validation_losses, label="Validation Loss", marker='o')
        plt.title("Training & Validation Loss Across K-Folds")
        plt.xlabel("Fold Number")
        plt.ylabel("Loss")
        plt.legend()
        plt.savefig(img_loss_path)
        plt.close()
        print(f"📉 Loss plot saved to {img_loss_path}")

    return model_algorithm

def get_training_loss(model, X_train, y_train):
    """
    Compute training loss based on model type.

    Args:
        model (object): Trained model.
        X_train (pd.DataFrame): Training feature matrix.
        y_train (pd.Series): Training labels.

    Returns:
        float: Training loss.
    """
    # Models that expose their loss during training
    if hasattr(model, "best_score_"):  # XGBoost
        return -model.best_score_

    if hasattr(model, "loss_"):  # Perceptron (Hinge loss)
        return model.loss_

    # Probabilistic models (e.g., HMM, Naive Bayes)
    if hasattr(model, "score"):  
        return -model.score(X_train, y_train)  # Negative log-likelihood

    # Support Vector Machines (hinge loss)
    if isinstance(model, SVC):
        y_pred = model.decision_function(X_train)
        return np.mean(np.maximum(0, 1 - y_train * y_pred))  # Hinge loss

    # Logistic Regression (log loss)
    if isinstance(model, LogisticRegression):
        y_proba = model.predict_proba(X_train)
        return log_loss(y_train, y_proba)

    # Decision Tree, Random Forest: No direct loss, use log loss
    if isinstance(model, (DecisionTreeClassifier, RandomForestClassifier)):
        y_proba = model.predict_proba(X_train)
        return log_loss(y_train, y_proba)

    return None  # Loss not available

# --------------------------------------------------
# Hung defined
def train_bayes_net(df, model_save_path):
    """
    Trains a Bayesian Network on the given DataFrame.

    Args:
        df (pd.DataFrame): Input DataFrame containing text data and target labels.
        model_save_path (str): Path to save the trained model.

    Returns:
        None
    """
    if os.path.exists(model_save_path):
        print("✅ Model found! Loading...")
        # reader = BIFReader(model_save_path)
        # model = reader.get_model()
        print("✅ Model loaded successfully!")
    else:  
        df_sampled = df
        
        vectorizer = CountVectorizer(binary=True, max_features=100) 
        X = vectorizer.fit_transform(df_sampled['text_clean']).toarray()
        y = df_sampled['target'].values
        
        # Chia dữ liệu thành tập huấn luyện và kiểm tra
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        
        # Chuyển đổi thành DataFrame để sử dụng với pgmpy
        feature_names = vectorizer.get_feature_names_out()
        train_df = pd.DataFrame(X_train, columns=feature_names)
        train_df['target'] = y_train
        
        # Xây dựng cấu trúc Bayesian Network
        # Giả sử mỗi từ phụ thuộc vào 'Sentiment'
        edges = [('target', word) for word in feature_names]
        model = BayesianNetwork(edges)
        
        # Học các bảng xác suất có điều kiện (CPT) từ dữ liệu
        model.fit(train_df, estimator=MaximumLikelihoodEstimator)

        # Suy luận và đánh giá mô hình
        inference = VariableElimination(model)
        
        # joblib.dump(model, model_save_path)
        # with open(model_save_path, "w") as f:
        #     f.write(model.to_bif())
        
        # print(f'💾 Model saved to {model_save_path}')

        # Hàm dự đoán sentiment cho tập dữ liệu
        def predict_sentiment(model, inference, X, feature_names):
            predictions = []
            for i in range(X.shape[0]):
                evidence = {feature_names[j]: X[i, j] for j in range(len(feature_names))}
                result = inference.map_query(variables=['target'], evidence=evidence)
                predictions.append(result['target'])
            return np.array(predictions)

        def predict_sentiment_proba(model, inference, X, feature_names):
            proba_predictions = []
            for i in range(X.shape[0]):
                evidence = {feature_names[j]: X[i, j] for j in range(len(feature_names))}
                result = inference.query(variables=['target'], evidence=evidence)
                
                # Extract probability of target = 1 (assuming binary classification: 0 or 1)
                prob_1 = result.values[1]  # Probabilities are stored as an array, index 1 corresponds to class 1
                proba_predictions.append(prob_1)
            return np.array(proba_predictions)
        
        # Dự đoán trên tập kiểm tra
        y_pred = predict_sentiment(model, inference, X_test, feature_names)

        # Đánh giá độ chính xác
        # Compute metrics
        accuracy = accuracy_score(y_test, y_pred)
        precision = precision_score(y_test, y_pred, average='binary')
        recall = recall_score(y_test, y_pred, average='binary')
        f1 = f1_score(y_test, y_pred, average='binary')
        
        # ROC AUC can be computed if the model outputs probabilities
        # Handle models that do not support `predict_proba`
        if hasattr(model, "predict_proba"):
            print("Has predict_proba")
            y_prob = model.predict_proba(X_test)[:, 1]  # Take the positive class probabilities
            roc_auc = roc_auc_score(y_test, y_prob)
        elif hasattr(model, "decision_function"):
            print("Has decision_function")
            y_prob = model.decision_function(X_test)
            roc_auc = roc_auc_score(y_test, y_prob)
        else:
            print("Does not have predict_proba or decision_function")
            y_proba = predict_sentiment_proba(model, inference, X_test, feature_names)
            roc_auc = roc_auc_score(y_test, y_proba)
            # roc_auc = "N/A"  # Not applicable for models like Perceptron

        # Print metrics
        print("Model: Bayesian Network")
        print("-" * 50)
        print(f"Accuracy: {accuracy:.4f}")
        print(f"Precision: {precision:.4f}")
        print(f"Recall: {recall:.4f}")
        print(f"F1 Score: {f1:.4f}")
        if hasattr(model, "predict_proba") or hasattr(model, "decision_function"):
            print(f"ROC AUC: {roc_auc:.4f}")
        else:
            if roc_auc != "N/A":
                print(f"ROC AUC: {roc_auc:.4f}")
            else:
                print("ROC AUC: N/A")

def extract_features(text, word_features):
    """
    Extracts features from text based on a given vocabulary.

    Args:
        text (str): Input text.
        word_features (list): List of word features.

    Returns:
        np.ndarray: Array of feature indices.
    """
    words = text.split()  # Chuyển văn bản thành danh sách từ
    return np.array([word_features.index(word) for word in words if word in word_features])

def pad_sequence(seq, max_len):
    """
    Pads a sequence to a fixed length.

    Args:
        seq (np.ndarray): Input sequence.
        max_len (int): Maximum length for padding.

    Returns:
        np.ndarray: Padded sequence.
    """
    if len(seq) >= max_len:
        return seq[:max_len]
    return np.pad(seq, (0, max_len - len(seq)), mode='constant', constant_values=0)

def train_hmm(df, model_save_path):
    """
    Trains a Hidden Markov Model (HMM) on the given DataFrame.

    Args:
        df (pd.DataFrame): Input DataFrame containing text data and target labels.
        model_save_path (str): Path to save the trained model.

    Returns:
        None
    """
    df_sampled = df
    
    # Tạo tập từ vựng (chỉ lấy 3000 từ phổ biến nhất)
    all_words = nltk.FreqDist(word.lower() for text in df_sampled["text_clean"] for word in text.split())
    word_features = list(all_words.keys())[:5000]  # Lấy 3000 từ phổ biến nhất

    # Chuyển đổi dữ liệu text thành dạng số
    X = [extract_features(text, word_features) for text in df_sampled["text_clean"]]
    y = df_sampled["target"].values  # Nhãn (0: negative, 1: positive)
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # HMM yêu cầu chuỗi có độ dài giống nhau -> Padding độ dài cố định (50 từ)
    max_len = 50
    
    X_train = np.array([pad_sequence(seq, max_len) for seq in X_train])
    X_test = np.array([pad_sequence(seq, max_len) for seq in X_test])

    # Huấn luyện HMM cho từng class (pos và neg)
    hmm = hmmlearn.hmm.GaussianHMM(n_components=2, covariance_type="diag", n_iter=100)
    hmm.fit(X_train)
    
    joblib.dump(hmm, model_save_path)
    print(f'💾 Model saved to {model_save_path}')
    
    y_pred = hmm.predict(X_test)
    
    # Compute metrics
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred, average='binary')
    recall = recall_score(y_test, y_pred, average='binary')
    f1 = f1_score(y_test, y_pred, average='binary')
    
    # ROC AUC can be computed if the model outputs probabilities
    # Handle models that do not support `predict_proba`
    if hasattr(hmm, "predict_proba"):
        print("Has predict_proba")
        y_prob = hmm.predict_proba(X_test)[:, 1]  # Take the positive class probabilities
        roc_auc = roc_auc_score(y_test, y_prob)
    elif hasattr(hmm, "decision_function"):
        print("Has decision_function")
        y_prob = hmm.decision_function(X_test)
        roc_auc = roc_auc_score(y_test, y_prob)
    else:
        print("Does not have predict_proba or decision_function")
        roc_auc = "N/A"  # Not applicable for models like Perceptron

    # Print metrics
    print("Model: HMM")
    print("-" * 50)
    print(f"Accuracy: {accuracy:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall: {recall:.4f}")
    print(f"F1 Score: {f1:.4f}")
    if hasattr(hmm, "predict_proba") or hasattr(hmm, "decision_function"):
        print(f"ROC AUC: {roc_auc:.4f}")
    else:
        print("ROC AUC: N/A")
    
def train_graphical_model(df, model_name, model_save_path):
    """
    Trains a graphical model (HMM or Bayesian Network) on the given DataFrame.

    Args:
        df (pd.DataFrame): Input DataFrame containing text data and target labels.
        model_name (str): Name of the model to train ("hmm" or "bayesnet").
        model_save_path (str): Path to save the trained model.

    Returns:
        None
    """
    if model_name == "hmm":
        train_hmm(df, model_save_path)
    elif model_name == "bayesnet":
        train_bayes_net(df, model_save_path)

# --------------------------------------------------

def train_cnn_lstm(texts, labels, vocab_size=10000, max_length=500, embedding_dim=100, num_trials=5, epochs=30):
    """
    Trains a CNN-LSTM sentiment analysis model on given text data.

    Args:
        texts (list): List of sentences (raw text).
        labels (list): List of binary sentiment labels (0 for negative, 1 for positive).
        vocab_size (int): Size of vocabulary for tokenization. Defaults to 10000.
        max_length (int): Maximum sequence length for padding. Defaults to 500.
        embedding_dim (int): Dimension of the word embedding layer. Defaults to 100.
        num_trials (int): Number of hyperparameter tuning trials. Defaults to 5.
        epochs (int): Number of training epochs. Defaults to 10.

    Returns:
        keras.Sequential: Trained Keras model with the best hyperparameters.
        dict: Dictionary containing training and validation metrics.
    """
    # **Step 1: Text Preprocessing**
    tokenizer = Tokenizer(num_words=vocab_size, oov_token="<OOV>")
    tokenizer.fit_on_texts(texts)
    sequences = tokenizer.texts_to_sequences(texts)
    X_data = pad_sequences(sequences, maxlen=max_length, padding="pre")
    y_data = np.array(labels)  # Convert labels to NumPy array

    # **Step 2: Split Data for Training & Testing**
    X_train, X_test, y_train, y_test = train_test_split(X_data, y_data, test_size=0.2, random_state=42)

    # **Step 3: Build Model Function**
    def build_model(hp):
        model = keras.Sequential()

        # **Embedding Layer**
        model.add(layers.Embedding(
            input_dim=vocab_size, 
            output_dim=embedding_dim, 
            input_length=max_length
        ))

        # **CNN Block 1**
        model.add(layers.Conv1D(
            filters=hp.Int('filters_1', min_value=64, max_value=256, step=64),
            kernel_size=hp.Choice('kernel_size_1', values=[3, 5, 7]),
            activation="relu",
            padding="same"
        ))
        model.add(layers.BatchNormalization())
        model.add(layers.MaxPooling1D(pool_size=2))

        # **CNN Block 2**
        model.add(layers.Conv1D(
            filters=hp.Int('filters_2', min_value=128, max_value=512, step=128),
            kernel_size=hp.Choice('kernel_size_2', values=[3, 5]),
            activation="relu",
            padding="same"
        ))
        model.add(layers.BatchNormalization())
        model.add(layers.MaxPooling1D(pool_size=2))

        # **Bidirectional LSTM Layer**
        model.add(layers.Bidirectional(layers.LSTM(
            units=hp.Int('lstm_units', min_value=64, max_value=256, step=64),
            activation="tanh",
            return_sequences=False
        )))

        # **Fully Connected Layer**
        model.add(layers.Dense(
            units=hp.Int('dense_units', min_value=128, max_value=512, step=128),
            activation="relu"
        ))
        model.add(layers.Dropout(rate=hp.Float('dropout', min_value=0.3, max_value=0.6, step=0.1)))

        # **Output Layer**
        model.add(layers.Dense(1, activation="sigmoid"))

        # **Compile Model**
        model.compile(
            optimizer=keras.optimizers.Adam(hp.Choice('learning_rate', values=[5e-4, 1e-4, 5e-5, 1e-5])),
            loss="binary_crossentropy",
            metrics=["accuracy"]
        )

        return model

    # **Step 4: Initialize Keras Tuner**
    tuner = kt.RandomSearch(
        build_model,
        objective="val_accuracy",
        max_trials=num_trials,
        executions_per_trial=1,
        directory="tuner_results",
        project_name="cnn_lstm_tuning"
    )

    print("\n🔍 Running Hyperparameter Tuning...")
    tuner.search(X_train, y_train, epochs=5, validation_data=(X_test, y_test), batch_size=32, validation_split=0.2, verbose=1)

    # **Step 5: Retrieve Best Model**
    best_hps = tuner.get_best_hyperparameters(num_trials=1)[0]
    best_model = tuner.hypermodel.build(best_hps)

    # **Step 6: Final Training with Best Model**
    print("\n🚀 Training Final Model...")
    history = best_model.fit(X_train, y_train, epochs=epochs, validation_data=(X_test, y_test), batch_size=32, validation_split=0.2, verbose=1)
    
    # **Step 7: Predict on Validation Set**
    y_pred_prob = best_model.predict(X_test)
    y_pred = (y_pred_prob > 0.5).astype(int)

    # **Step 8: Compute Metrics**
    precision = precision_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    roc_auc = roc_auc_score(y_test, y_pred_prob)

    # **Step 9: Store Results**
    results = {
        "loss": history.history["loss"],
        "val_loss": history.history["val_loss"],
        "accuracy": history.history["accuracy"],
        "val_accuracy": history.history["val_accuracy"],
        "precision": precision,
        "recall": recall,
        "f1_score": f1,
        "roc_auc": roc_auc
    }

    print(f'🔹 loss: {history.history["loss"][-1]}')
    print(f'🔹 val_loss: {history.history["val_loss"][-1]}')
    print(f'🔹 accuracy: {history.history["accuracy"][-1]}')
    print(f'🔹 val_accuracy: {history.history["val_accuracy"][-1]}')
    print(f'🔹 precision: {precision}')
    print(f'🔹 recall: {recall}')
    print(f'🔹 f1_score: {f1}')
    print(f'🔹 roc_auc: {roc_auc}')

    # **Step 10: Save the Best Model**
    best_model.save("best_cnn_lstm.keras")

    print("\n✅ Model Training and Save Complete!")
    
    # === Plot Training & Validation Loss ===
    plt.figure(figsize=(10, 5))
    plt.plot(history.history["loss"], label="Training Loss")
    plt.plot(history.history["val_loss"], label="Validation Loss")
    plt.title("📉 Training and Validation Loss over Epochs")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig("cnn_lstm_loss.png")
    plt.close()

    # === Plot Training & Validation Accuracy ===
    plt.figure(figsize=(10, 5))
    plt.plot(history.history["accuracy"], label="Training Accuracy")
    plt.plot(history.history["val_accuracy"], label="Validation Accuracy")
    plt.title("📈 Training and Validation Accuracy over Epochs")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig("cnn_lstm_accuracy.png")
    plt.close()

    print("📊 Training curves saved: cnn_lstm_loss.png, cnn_lstm_accuracy.png")
    
    return best_model, results

def train_bilstm_model(texts, labels, vocab_size=10000, max_length=500, embedding_dim=100, epochs=30):
    tokenizer = Tokenizer(num_words=vocab_size, oov_token="<OOV>")
    tokenizer.fit_on_texts(texts)
    sequences = tokenizer.texts_to_sequences(texts)
    X_data = pad_sequences(sequences, maxlen=max_length)
    y_data = np.array(labels)

    X_train, X_test, y_train, y_test = train_test_split(X_data, y_data, test_size=0.2, random_state=42)

    model = keras.Sequential([
        layers.Embedding(input_dim=vocab_size, output_dim=embedding_dim, input_length=max_length),
        layers.Bidirectional(layers.LSTM(128, return_sequences=False)),
        layers.Dense(128, activation="relu"),
        layers.Dropout(0.5),
        layers.Dense(1, activation="sigmoid")
    ])

    model.compile(optimizer="adam", loss="binary_crossentropy", metrics=["accuracy"])

    print("\n🚀 Training Bi-LSTM model...")
    history = model.fit(X_train, y_train, validation_data=(X_test, y_test), batch_size=32, epochs=epochs, verbose=1)

    y_pred_prob = model.predict(X_test)
    y_pred = (y_pred_prob > 0.5).astype(int)

    precision = precision_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    roc_auc = roc_auc_score(y_test, y_pred_prob)

    results = {
        "loss": history.history["loss"],
        "val_loss": history.history["val_loss"],
        "accuracy": history.history["accuracy"],
        "val_accuracy": history.history["val_accuracy"],
        "precision": precision,
        "recall": recall,
        "f1_score": f1,
        "roc_auc": roc_auc
    }

    print(f'🔹 loss: {history.history["loss"][-1]}')
    print(f'🔹 val_loss: {history.history["val_loss"][-1]}')
    print(f'🔹 accuracy: {history.history["accuracy"][-1]}')
    print(f'🔹 val_accuracy: {history.history["val_accuracy"][-1]}')
    print(f'🔹 precision: {precision}')
    print(f'🔹 recall: {recall}')
    print(f'🔹 f1_score: {f1}')
    print(f'🔹 roc_auc: {roc_auc}')

    model.save("best_bilstm_model.keras")

    # Plot Loss
    plt.figure(figsize=(10, 5))
    plt.plot(history.history["loss"], label="Training Loss")
    plt.plot(history.history["val_loss"], label="Validation Loss")
    plt.title("📉 Bi-LSTM Training and Validation Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend()
    plt.tight_layout()
    plt.grid(True)
    plt.savefig("bilstm_loss.png")
    plt.close()

    # Plot Accuracy
    plt.figure(figsize=(10, 5))
    plt.plot(history.history["accuracy"], label="Training Accuracy")
    plt.plot(history.history["val_accuracy"], label="Validation Accuracy")
    plt.title("📈 Bi-LSTM Training and Validation Accuracy")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.legend()
    plt.tight_layout()
    plt.grid(True)
    plt.savefig("bilstm_accuracy.png")
    plt.close()

    print("📊 Training curves saved: bilstm_loss.png, bilstm_accuracy.png")

    return model, results

def train_bert_model(texts, labels, model_name='bert-base-uncased', epochs=4, batch_size=32, max_length=128):
    # Step 1: Tokenization
    tokenizer = BertTokenizer.from_pretrained(model_name)

    def encode_texts(texts, labels):
        return tokenizer(
            texts,
            padding=True,
            truncation=True,
            max_length=max_length,
            return_tensors='tf'
        ), tf.convert_to_tensor(labels)

    X_encoded, y_encoded = encode_texts(texts, labels)

    # Step 2: Split dataset
    X_train, X_test, y_train, y_test = train_test_split(X_encoded['input_ids'], y_encoded, test_size=0.2, random_state=42)

    # Step 3: Model definition
    model = TFBertForSequenceClassification.from_pretrained(model_name, num_labels=2)

    optimizer = keras.optimizers.Adam(learning_rate=2e-5)
    loss = keras.losses.SparseCategoricalCrossentropy(from_logits=True)
    metrics = ['accuracy']

    model.compile(optimizer=optimizer, loss=loss, metrics=metrics)

    # Step 4: Train the model
    print("\n🚀 Training BERT model...")
    history = model.fit(
        X_train, y_train,
        validation_data=(X_test, y_test),
        batch_size=batch_size,
        epochs=epochs,
        verbose=1
    )

    # Step 5: Inference
    y_pred_logits = model.predict(X_test).logits
    y_pred = np.argmax(y_pred_logits, axis=1)

    # Step 6: Metrics
    precision = precision_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    roc_auc = roc_auc_score(y_test, tf.nn.softmax(y_pred_logits)[:, 1])

    results = {
        "loss": history.history["loss"],
        "val_loss": history.history["val_loss"],
        "accuracy": history.history["accuracy"],
        "val_accuracy": history.history["val_accuracy"],
        "precision": precision,
        "recall": recall,
        "f1_score": f1,
        "roc_auc": roc_auc
    }

    # Step 7: Save
    model.save_pretrained("best_bert_model")

    print("\n📊 Metrics:")
    print(f'🔹 Precision: {precision}')
    print(f'🔹 Recall: {recall}')
    print(f'🔹 F1-Score: {f1}')
    print(f'🔹 ROC AUC: {roc_auc}')

    # Plot Loss
    plt.figure(figsize=(10, 5))
    plt.plot(history.history["loss"], label="Training Loss")
    plt.plot(history.history["val_loss"], label="Validation Loss")
    plt.title("📉 BERT Training and Validation Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend()
    plt.tight_layout()
    plt.grid(True)
    plt.savefig("bert_loss.png")
    plt.close()

    # Plot Accuracy
    plt.figure(figsize=(10, 5))
    plt.plot(history.history["accuracy"], label="Training Accuracy")
    plt.plot(history.history["val_accuracy"], label="Validation Accuracy")
    plt.title("📈 BERT Training and Validation Accuracy")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.legend()
    plt.tight_layout()
    plt.grid(True)
    plt.savefig("bert_accuracy.png")
    plt.close()

    print("✅ Model saved. Training curves saved as bert_loss.png and bert_accuracy.png")

    return model, results

# --------------------------------------------------

from sklearn.metrics import precision_score, recall_score, f1_score, roc_auc_score
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import os
# from keras.preprocessing.text import Tokenizer
# from keras.utils import pad_sequences
import optuna
from torch.utils.data import Dataset, DataLoader
import json

START_TAG = "<START>"
STOP_TAG = "<STOP>"
tag_to_ix = {START_TAG: 0, STOP_TAG: 1, "NEG": 2, "POS": 3}
ix_to_tag = {v: k for k, v in tag_to_ix.items()}

def argmax(vec):
    return torch.argmax(vec)

def log_sum_exp(vec):
    max_score = vec.max()
    max_score_broadcast = max_score.view(1, -1).expand(1, vec.size()[1])
    return max_score + torch.log(torch.sum(torch.exp(vec - max_score_broadcast)))

class BiLSTM_CRF_FeatureExtractor(nn.Module):
    def __init__(self, vocab_size, tag_to_ix, embedding_dim, hidden_dim):
        super(BiLSTM_CRF_FeatureExtractor, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        self.lstm = nn.LSTM(embedding_dim, hidden_dim // 2,
                            num_layers=1, bidirectional=True, batch_first=True)
        self.hidden2tag = nn.Linear(hidden_dim, len(tag_to_ix))
        self.transitions = nn.Parameter(torch.randn(len(tag_to_ix), len(tag_to_ix)))
        self.tag_to_ix = tag_to_ix

        self.transitions.data[tag_to_ix[START_TAG], :] = -10000.
        self.transitions.data[:, tag_to_ix[STOP_TAG]] = -10000.

    def _get_lstm_features(self, sentence):
        embeds = self.embedding(sentence)
        lstm_out, _ = self.lstm(embeds)
        return self.hidden2tag(lstm_out)

    def _forward_alg(self, feats):
        init_alphas = torch.full((1, len(self.tag_to_ix)), -10000., device=feats.device)
        init_alphas[0][self.tag_to_ix[START_TAG]] = 0.
        forward_var = init_alphas
        for feat in feats:
            alphas_t = []
            for next_tag in range(len(self.tag_to_ix)):
                emit_score = feat[next_tag].view(1, -1).expand(1, len(self.tag_to_ix))
                trans_score = self.transitions[next_tag].view(1, -1)
                next_tag_var = forward_var + trans_score + emit_score
                alphas_t.append(log_sum_exp(next_tag_var).view(1))
            forward_var = torch.cat(alphas_t).view(1, -1)
        terminal_var = forward_var + self.transitions[self.tag_to_ix[STOP_TAG]]
        return log_sum_exp(terminal_var)

    def _score_sentence(self, feats, tags):
        score = torch.zeros(1, device=feats.device)
        tags = torch.cat([torch.tensor([self.tag_to_ix[START_TAG]], device=feats.device), tags])
        for i, feat in enumerate(feats):
            score += self.transitions[tags[i + 1], tags[i]] + feat[tags[i + 1]]
        score += self.transitions[self.tag_to_ix[STOP_TAG], tags[-1]]
        return score

    def _viterbi_decode(self, feats):
        backpointers = []
        init_vvars = torch.full((1, len(self.tag_to_ix)), -10000., device=feats.device)
        init_vvars[0][self.tag_to_ix[START_TAG]] = 0
        forward_var = init_vvars

        for feat in feats:
            bptrs_t = []
            viterbivars_t = []
            for next_tag in range(len(self.tag_to_ix)):
                next_tag_var = forward_var + self.transitions[next_tag]
                best_tag_id = argmax(next_tag_var)
                bptrs_t.append(best_tag_id.item())
                viterbivars_t.append(next_tag_var[0][best_tag_id].view(1))
            forward_var = (torch.cat(viterbivars_t) + feat).view(1, -1)
            backpointers.append(bptrs_t)

        terminal_var = forward_var + self.transitions[self.tag_to_ix[STOP_TAG]]
        best_tag_id = argmax(terminal_var)
        best_path = [best_tag_id.item()]
        for bptrs_t in reversed(backpointers):
            best_tag_id = bptrs_t[best_tag_id]
            best_path.append(best_tag_id)
        start = best_path.pop()
        best_path.reverse()
        return best_path

    def neg_log_likelihood(self, sentences, tags):
        feats = self._get_lstm_features(sentences)
        forward_score = self._forward_alg(feats[0])
        gold_score = self._score_sentence(feats[0], tags[0])
        return forward_score - gold_score

    def forward(self, sentences):
        feats = self._get_lstm_features(sentences)
        return self._viterbi_decode(feats[0])

class CRFSentimentDataset(Dataset):
    def __init__(self, input_ids, labels, max_len):
        self.input_ids = input_ids
        self.labels = labels
        self.max_len = max_len

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        x = torch.tensor(self.input_ids[idx], dtype=torch.long)
        tag_label = "POS" if self.labels[idx] == 1 else "NEG"
        tag_id = tag_to_ix[tag_label]
        y = torch.tensor([tag_id] * self.max_len, dtype=torch.long)
        return x, y

def tag_sequence_to_sentiment(tag_seq):
    count_pos = tag_seq.count(tag_to_ix["POS"])
    count_neg = tag_seq.count(tag_to_ix["NEG"])
    return 1 if count_pos >= count_neg else 0

def train_crf_feature_extractor(texts, labels, vocab_size=10000, max_length=100, embedding_dim=100, num_trials=5, epochs=30):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"🖥️ Using device: {device}")

    tokenizer = Tokenizer(num_words=vocab_size, oov_token="<OOV>")
    tokenizer.fit_on_texts(texts)
    sequences = tokenizer.texts_to_sequences(texts)
    X_data = pad_sequences(sequences, maxlen=max_length, padding="post")
    y_data = np.array(labels)

    X_temp, X_test, y_temp, y_test = train_test_split(X_data, y_data, test_size=0.2, random_state=42)
    X_train, X_val, y_train, y_val = train_test_split(X_temp, y_temp, test_size=0.2, random_state=42)

    def objective(trial):
        hidden_dim = trial.suggest_int("hidden_dim", 64, 256, step=64)
        lr = trial.suggest_float("lr", 1e-5, 1e-3, log=True)

        model = BiLSTM_CRF_FeatureExtractor(vocab_size, tag_to_ix, embedding_dim, hidden_dim).to(device)
        optimizer = torch.optim.Adam(model.parameters(), lr=lr)

        train_loader = DataLoader(CRFSentimentDataset(X_train, y_train, max_length), batch_size=32, shuffle=True)
        val_loader = DataLoader(CRFSentimentDataset(X_val, y_val, max_length), batch_size=32)

        model.train()
        for _ in range(3):
            for x, y in train_loader:
                x, y = x.to(device), y.to(device)
                model.zero_grad()
                loss = model.neg_log_likelihood(x, y)
                loss.backward()
                optimizer.step()

        model.eval()
        y_pred, y_true = [], []
        with torch.no_grad():
            for x, y in val_loader:
                x = x.to(device)
                outputs = model(x)
                
                for i in range(x.size(0)):
                    decoded = model(x[i].unsqueeze(0))
                    y_pred.append(tag_sequence_to_sentiment(decoded))
                    y_true.append(1 if y[i][0].item() == tag_to_ix["POS"] else 0)

        return f1_score(y_true, y_pred)

    print("🔍 Tuning hyperparameters...")
    study = optuna.create_study(direction="maximize")
    study.optimize(objective, n_trials=num_trials)
    best_params = study.best_params

    model = BiLSTM_CRF_FeatureExtractor(vocab_size, tag_to_ix, embedding_dim, best_params["hidden_dim"]).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=best_params["lr"])
    train_loader = DataLoader(CRFSentimentDataset(X_train, y_train, max_length), batch_size=32, shuffle=True)
    val_loader = DataLoader(CRFSentimentDataset(X_val, y_val, max_length), batch_size=32)

    history = {"train_loss": [], "val_loss": []}

    for epoch in range(epochs):
        model.train()
        total_train_loss = 0.0
        for x, y in train_loader:
            x, y = x.to(device), y.to(device)
            model.zero_grad()
            loss = model.neg_log_likelihood(x, y)
            loss.backward()
            optimizer.step()
            total_train_loss += loss.item()
        avg_train_loss = total_train_loss / len(train_loader)

        model.eval()
        total_val_loss = 0.0
        with torch.no_grad():
            for x, y in val_loader:
                x, y = x.to(device), y.to(device)
                total_val_loss += model.neg_log_likelihood(x, y).item()
        avg_val_loss = total_val_loss / len(val_loader)

        history["train_loss"].append(avg_train_loss)
        history["val_loss"].append(avg_val_loss)
        print(f"Epoch {epoch+1}/{epochs} - Train Loss: {avg_train_loss:.4f}, Val Loss: {avg_val_loss:.4f}")

    test_loader = DataLoader(CRFSentimentDataset(X_test, y_test, max_length), batch_size=32)
    model.eval()
    y_pred, y_true = [], []
    with torch.no_grad():
        for x, y in test_loader:
            x = x.to(device)
            outputs = model(x)
            for i in range(x.size(0)):
                decoded = model(x[i].unsqueeze(0))
                y_pred.append(tag_sequence_to_sentiment(decoded))
                y_true.append(1 if y[i][0].item() == tag_to_ix["POS"] else 0)


    accuracy = accuracy_score(y_true, y_pred)
    precision = precision_score(y_true, y_pred)
    recall = recall_score(y_true, y_pred)
    f1 = f1_score(y_true, y_pred)
    roc_auc = roc_auc_score(y_true, y_pred)
    report = classification_report(y_true, y_pred, target_names=["Negative", "Positive"])
    print("\nClassification Report:\n", report)

    results = {
        "train_loss": history["train_loss"],
        "val_loss": history["val_loss"],
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "f1_score": f1,
        "roc_auc": roc_auc
    }

    print(f"\n✅ Test - Accuracy: {accuracy:.4f}, Precision: {precision:.4f}, Recall: {recall:.4f}, F1: {f1:.4f}, AUC: {roc_auc:.4f}")

    os.makedirs("crf_feature_model", exist_ok=True)
    torch.save(model.state_dict(), "best_crf_model.pt")
    with open("best_crf_model_config.json", "w") as f:
        json.dump(best_params, f, indent=4)

    plt.figure(figsize=(10, 5))
    plt.plot(history["train_loss"], label="Train Loss")
    plt.plot(history["val_loss"], label="Validation Loss")
    plt.title("📉 Train vs Validation Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig("crf_loss_curve.png")
    plt.close()

    return model, results


# --------------------------------------------------

def train_general_model(df, doc_lst, label_lst, model_name_lst, feature_methods, model_dict, param_dict, X_train_features_dict, X_test_features_dict, y_train, y_test):
    """
    Trains general models using specified feature extraction methods and model algorithms.

    Args:
        df (pd.DataFrame): Input DataFrame containing text data and target labels.
        doc_lst (list): List of documents (text data).
        label_lst (list): List of labels corresponding to the documents.
        model_name_lst (list): List of model names to train.
        feature_methods (list): List of feature extraction methods to use.
        model_dict (dict): Dictionary mapping model names to model classes.
        param_dict (dict): Dictionary mapping model names to hyperparameter grids.
        X_train_features_dict (dict): Dictionary of training feature matrices for each method.
        X_test_features_dict (dict): Dictionary of testing feature matrices for each method.
        y_train (pd.Series): Training labels.
        y_test (pd.Series): Testing labels.

    Returns:
        None
    """
    print("\n🔎 Running feature extraction and model training loop...\n")
    
    for model_name in model_name_lst:
        print(f"\n🚀 Training {model_name} models...\n")

        try:
            if model_name == "cnn" or model_name == "lstm":
                train_cnn_lstm(doc_lst, label_lst)
                
            elif model_name == "CRF":
                train_crf_feature_extractor(doc_lst, label_lst)
                
            elif model_name == "bilstm":
                train_bilstm_model(doc_lst, label_lst)
                
            elif model_name == "bert":
                train_bert_model(doc_lst, label_lst)
                
            elif model_name == "hmm" or model_name == "bayesnet":
                train_graphical_model(
                    df, 
                    model_name, 
                    model_save_path=f"best_{model_name}.pkl"
                )
                
            else:
                for method in feature_methods:
                    print(f"🔎 Training with Method: {method}...")
                    
                    if model_name == "GA":
                        genetic_algorithm(
                            X_train_features_dict[method], 
                            y_train, 
                            X_test_features_dict[method], 
                            y_test, 
                            model_save_path=f"best_{model_name}_{method}.pkl",
                            img_save_path=f"best_{model_name}_{method}.png",
                            img_loss_path=f"best_{model_name}_{method}_loss.png"
                        )
                    
                    else:
                        model_api = model_dict[model_name]()
                        model_params = param_dict[model_name]
                        
                        generate_binary_classification_model(
                            X=X_train_features_dict[method], 
                            y=y_train, 
                            model_algorithm=model_api, 
                            hyperparameters=model_params, 
                            needs_scaled=False, 
                            model_save_path=f"best_{model_name}_{method}.pkl",
                            img_save_path=f"best_{model_name}_{method}.png",
                            img_loss_path=f"best_{model_name}_{method}_loss.png"
                        )
                        
        except Exception as e:
            print(f"❌ Error with {model_name}: {e}")

# --------------------------------------------------  

def predict_general_model(model_names, feature_methods, X_test_features_dict, y_test, output_dir):
    """
    Predicts using trained models and evaluates their performance.

    Args:
        model_names (list): List of model names to use for prediction.
        feature_methods (list): List of feature extraction methods to use.
        X_test_features_dict (dict): Dictionary of testing feature matrices for each method.
        y_test (pd.Series): Testing labels.
        output_dir (str): Directory to save the prediction results.

    Returns:
        None
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"⚙️  Using device: {device}")

    for model_name in model_names:
        if model_name in ["GA", "hmm", "bayesnet", "lstm", "CRF"]:
            print(f"Already trained and tested model: {model_name}")
            continue

        for method in feature_methods:
            print(f"🔎 Predicting with Model: {model_name}, Method: {method}...")

            try:
                if model_name == "cnn":
                    model_filename = os.path.join(output_dir, f"best_{model_name}.keras")
                    model = tf.keras.models.load_model(model_filename)

                    X_test_features = np.array(X_test_features_dict[method])
                    input_shape = (X_test_features.shape[1], 1)
                    X_test_features = X_test_features.reshape(-1, X_test_features.shape[1], 1)

                    y_prob = model.predict(X_test_features).flatten()
                    y_pred = (y_prob > 0.5).astype(int)

                else:  # Handle Machine Learning models
                    # Load the saved model
                    model_filename = os.path.join(output_dir, f"best_{model_name}_{method}.pkl")
                    with open(model_filename, 'rb') as model_file:
                        model = joblib.load(model_file)

                    # Make predictions
                    y_pred = model.predict(X_test_features_dict[method])
                    
                    # ROC AUC can be computed if the model outputs probabilities
                    # Handle models that do not support `predict_proba`
                    if hasattr(model, "predict_proba"):
                        y_prob = model.predict_proba(X_test_features_dict[method])[:, 1]  # Take the positive class probabilities
                    elif hasattr(model, "decision_function"):
                        y_prob = model.decision_function(X_test_features_dict[method])
                    else:
                        y_prob = None

                # Compute metrics
                accuracy = accuracy_score(y_test, y_pred)
                precision = precision_score(y_test, y_pred, average='binary')
                recall = recall_score(y_test, y_pred, average='binary')
                f1 = f1_score(y_test, y_pred, average='binary')
                roc_auc = roc_auc_score(y_test, y_prob) if y_prob is not None else "N/A"

                print(f"Model: {model_name}")
                print(f"Method: {method}")
                print("-" * 50)
                print(f"Accuracy: {accuracy:.4f}")
                print(f"Precision: {precision:.4f}")
                print(f"Recall: {recall:.4f}")
                print(f"F1 Score: {f1:.4f}")
                print(f"ROC AUC: {roc_auc if roc_auc != 'N/A' else 'N/A'}")
                # Print classification report for binary classification (0 = negative, 1 = positive)
                print("\n🔬 Classification Report:")
                print(classification_report(y_test, y_pred, labels=[0, 1], target_names=["negative", "positive"]))

                    
            except Exception as e:
                print(f"❌ Error while predicting for {model_name} with {method}: {e}")

            
        print("%" * 50)
        print("%" * 50)

# --------------------------------------------------
# helper plot func
def plot_results(accuracy, roc_auc, train_loss, val_loss, img_save_path, img_loss_path):
    if img_save_path:
        plt.figure()
        plt.plot(accuracy, label="Accuracy", marker='o')
        plt.plot(roc_auc, label="ROC AUC", marker='o')
        plt.legend()
        plt.title("Validation Performance")
        plt.savefig(img_save_path)
        print(f"📈 Performance plot saved to {img_save_path}")

    if img_loss_path:
        plt.figure()
        plt.plot(train_loss, label="Train Loss", marker='o')
        plt.plot(val_loss, label="Val Loss", marker='o')
        plt.legend()
        plt.title("Loss Curves")
        plt.savefig(img_loss_path)
        print(f"📉 Loss plot saved to {img_loss_path}")

# Voting - test ok
def train_voting_classifier(model_dict, param_dict, feature_method, X, y, voting_type='soft', model_save_path="voting_model.pkl", img_save_path=None, img_loss_path=None):
    """
    Trains a Voting Classifier using selected models with cross-validation.
    
    Args:
        model_dict (dict): Dictionary of models.
        param_dict (dict): Dictionary of best hyperparameters.
        feature_method (str): Name of feature extraction method used.
        X (array-like): Feature matrix.
        y (array-like): Labels.
        voting_type (str): 'hard' for majority vote, 'soft' for probability-based averaging.
        model_save_path (str): Path to save the trained model.
        img_save_path (str, optional): Path to save validation performance plot.
        img_loss_path (str, optional): Path to save training loss plot.

    Returns:
        VotingClassifier model.
    """

    # Load existing model if available
    if os.path.exists(model_save_path):
        print(f"🔄 Loading existing model from {model_save_path}...")
        return joblib.load(model_save_path)

    print(f"\n🚀 Training Voting Classifier ({voting_type}) with feature method: {feature_method}\n")

    # Create base models with their best parameters
    base_models = []
    for model_name in model_dict.keys():
        try:
            model = model_dict[model_name](**param_dict.get(model_name, {}))  # Use best params
            base_models.append((model_name, model))
        except Exception as e:
            print(f"⚠️ Skipping {model_name} due to error: {e}")

    # Ensure at least 2 models exist
    if len(base_models) < 2:
        print("❌ Not enough models to perform voting.")
        return None

    # Define VotingClassifier
    voting_clf = VotingClassifier(estimators=base_models, voting=voting_type)
    
    # Cross-validation
    accuracy_scores, roc_auc_scores, f1_scores, precision_scores, recall_scores = [], [], [], [], []
    training_losses, validation_losses = [], []

    k_fold = KFold(n_splits=5)
    print("\n🎯 Running K-Fold Cross-Validation...")
    for train_idx, val_idx in tqdm(k_fold.split(X), total=k_fold.get_n_splits(), desc="K-Fold Progress"):
        X_train, X_val = X.iloc[train_idx, :], X.iloc[val_idx, :]
        y_train, y_val = y.iloc[train_idx], y.iloc[val_idx]
        
        voting_clf.fit(X_train, y_train)

        train_loss = get_training_loss(voting_clf, X_train, y_train)
        val_loss = get_training_loss(voting_clf, X_val, y_val)

        training_losses.append(train_loss)
        validation_losses.append(val_loss)

        val_preds = voting_clf.predict(X_val)

        accuracy_scores.append(accuracy_score(y_val, val_preds))
        roc_auc_scores.append(roc_auc_score(y_val, val_preds))
        f1_scores.append(f1_score(y_val, val_preds))
        precision_scores.append(precision_score(y_val, val_preds))
        recall_scores.append(recall_score(y_val, val_preds))

    # Print results
    print(f'📊 Avg Accuracy: {mean(accuracy_scores):.4f}')
    print(f'📊 Avg ROC AUC: {mean(roc_auc_scores):.4f}')
    print(f'📊 Avg F1 Score: {mean(f1_scores):.4f}')
    print(f'📊 Avg Precision: {mean(precision_scores):.4f}')
    print(f'📊 Avg Recall: {mean(recall_scores):.4f}')

    # Train on full dataset
    voting_clf.fit(X, y)
    joblib.dump(voting_clf, model_save_path)
    print(f'💾 Model saved to {model_save_path}')

    # Plot performance & loss curves
    plot_results(accuracy_scores, roc_auc_scores, training_losses, validation_losses, img_save_path, img_loss_path)

    return voting_clf

# Stacking - test ok
def train_stacking_classifier(model_dict, param_dict, feature_method, X, y, final_estimator=LogisticRegression(), model_save_path="stacking_model.pkl", img_save_path=None, img_loss_path=None):
    """
    Trains a Stacking Classifier using selected models with cross-validation.
    
    Args:
        model_dict (dict): Dictionary of models.
        param_dict (dict): Dictionary of best hyperparameters.
        feature_method (str): Name of feature extraction method used.
        X (array-like): Feature matrix.
        y (array-like): Labels.
        final_estimator (sklearn model): Meta-model for final prediction (default: LogisticRegression).
        model_save_path (str): Path to save the trained model.
        img_save_path (str, optional): Path to save validation performance plot.
        img_loss_path (str, optional): Path to save training loss plot.

    Returns:
        StackingClassifier model.
    """

    if os.path.exists(model_save_path):
        print(f"🔄 Loading existing model from {model_save_path}...")
        return joblib.load(model_save_path)

    print(f"\n🚀 Training Stacking Classifier with feature method: {feature_method}\n")

    base_models = []
    for model_name in model_dict.keys():
        try:
            model = model_dict[model_name](**param_dict.get(model_name, {}))
            base_models.append((model_name, model))
        except Exception as e:
            print(f"⚠️ Skipping {model_name} due to error: {e}")

    if len(base_models) < 2:
        print("❌ Not enough models to perform stacking.")
        return None

    stacking_clf = StackingClassifier(estimators=base_models, final_estimator=final_estimator)
    
    # Cross-validation
    accuracy_scores, roc_auc_scores, f1_scores, precision_scores, recall_scores = [], [], [], [], []
    training_losses, validation_losses = [], []
    
    k_fold = KFold(n_splits=5)
    print("\n🎯 Running K-Fold Cross-Validation...")
    for train_idx, val_idx in tqdm(k_fold.split(X), total=k_fold.get_n_splits(), desc="K-Fold Progress"):
        X_train, X_val = X.iloc[train_idx, :], X.iloc[val_idx, :]
        y_train, y_val = y.iloc[train_idx], y.iloc[val_idx]

        stacking_clf.fit(X_train, y_train)

        train_loss = get_training_loss(stacking_clf, X_train, y_train)
        val_loss = get_training_loss(stacking_clf, X_val, y_val)

        training_losses.append(train_loss)
        validation_losses.append(val_loss)

        val_preds = stacking_clf.predict(X_val)
        
        accuracy_scores.append(accuracy_score(y_val, val_preds))
        roc_auc_scores.append(roc_auc_score(y_val, val_preds))
        f1_scores.append(f1_score(y_val, val_preds))
        precision_scores.append(precision_score(y_val, val_preds))
        recall_scores.append(recall_score(y_val, val_preds))

    # Print results
    print(f'📊 Avg Accuracy: {mean(accuracy_scores):.4f}')
    print(f'📊 Avg ROC AUC: {mean(roc_auc_scores):.4f}')
    print(f'📊 Avg F1 Score: {mean(f1_scores):.4f}')
    print(f'📊 Avg Precision: {mean(precision_scores):.4f}')
    print(f'📊 Avg Recall: {mean(recall_scores):.4f}')
    
    stacking_clf.fit(X, y)
    joblib.dump(stacking_clf, model_save_path)
    print(f'💾 Model saved to {model_save_path}')

    plot_results(accuracy_scores, roc_auc_scores, training_losses, validation_losses, img_save_path, img_loss_path)

    return stacking_clf


