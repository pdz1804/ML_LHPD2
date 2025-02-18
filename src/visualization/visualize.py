"""
visualize.py

A module for providing various data visualization options for any dataset.
This includes plotting class distributions, feature relationships, correlation matrices,
and automated visualization of all columns and relationships.

STILL NOT BE ABLE TO USE THIS YET :>

Author: Nguyen Quang Phu

Date: 2025-01-21

Usage:
    from visualization import DataVisualizer
    visualizer = DataVisualizer(data)
    visualizer.plot_class_distribution("target")
"""

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

class DataVisualizer:
    """
    A class to provide various data visualization options for any dataset.
    """

    def __init__(self, data):
        """
        Initialize the visualizer with a dataset.
        :param data: pandas DataFrame containing the dataset.
        """
        if not isinstance(data, pd.DataFrame):
            raise ValueError("The data should be a pandas DataFrame.")
        self.data = data

    def plot_overview(self):
        """
        Provide a general overview of the dataset, including:
        - Number of rows and columns
        - Sample data
        - Column types and null values
        """
        print("Dataset Overview:")
        print(f"Number of samples (rows): {self.data.shape[0]}")
        print(f"Number of features (columns): {self.data.shape[1]}")
        print("\nColumn Data Types:")
        print(self.data.dtypes)
        print("\nNull Values per Column:")
        print(self.data.isnull().sum())
        print("\nSample Data:")
        print(self.data.head())

    def plot_class_distribution(self, column):
        """
        Plot the distribution of a specific column.
        :param column: Column name to visualize.
        """
        if column not in self.data.columns:
            raise ValueError(f"Column '{column}' does not exist in the dataset.")
        if self.data[column].dtype == "object" or self.data[column].nunique() <= 20:
            self.data[column].value_counts().plot(kind='bar', color='skyblue')
            plt.title(f"Distribution of {column}")
            plt.xlabel(column)
            plt.ylabel("Frequency")
            plt.show()
        else:
            sns.histplot(self.data[column], kde=True, bins=30, color="skyblue")
            plt.title(f"Distribution of {column}")
            plt.xlabel(column)
            plt.ylabel("Density")
            plt.show()

    def plot_feature_relationship(self, feature1, feature2):
        """
        Plot the relationship between two features.
        :param feature1: First feature for the x-axis.
        :param feature2: Second feature for the y-axis.
        """
        if feature1 not in self.data.columns or feature2 not in self.data.columns:
            raise ValueError(f"One or both features '{feature1}' and '{feature2}' do not exist in the dataset.")
        plt.figure(figsize=(8, 6))
        sns.scatterplot(x=self.data[feature1], y=self.data[feature2], alpha=0.7, color="blue")
        plt.title(f"Relationship between {feature1} and {feature2}")
        plt.xlabel(feature1)
        plt.ylabel(feature2)
        plt.show()

    def plot_correlation_matrix(self):
        """
        Plot the correlation matrix for numerical features in the dataset.
        """
        corr = self.data.corr()
        if corr.empty:
            print("No numerical features to compute correlation.")
            return
        plt.figure(figsize=(10, 8))
        sns.heatmap(corr, annot=True, fmt=".2f", cmap="coolwarm", square=True)
        plt.title("Correlation Matrix")
        plt.show()

    def plot_all_distributions(self):
        """
        Plot the distributions of all columns in the dataset.
        """
        for column in self.data.columns:
            print(f"Plotting distribution for: {column}")
            try:
                self.plot_class_distribution(column)
            except ValueError as e:
                print(f"Skipping {column}: {e}")

    def plot_all_relationships(self):
        """
        Plot scatter plots for all pairwise relationships between numerical columns.
        """
        numeric_cols = self.data.select_dtypes(include=["number"]).columns
        if len(numeric_cols) < 2:
            print("Not enough numerical features for pairwise relationships.")
            return
        sns.pairplot(self.data[numeric_cols], diag_kind="kde", plot_kws={"alpha": 0.7})
        plt.suptitle("Pairwise Relationships", y=1.02)
        plt.show()


