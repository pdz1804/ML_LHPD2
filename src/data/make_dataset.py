"""
make_dataset.py

A module for defining a Dataset class to visualize and explore datasets.

Author: Nguyen Quang Phu
Date: 2025-01-21
"""

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

class Dataset:
    """
    A class for dataset visualization and exploration.
    """

    def __init__(self, data):
        """
        Initialize the Dataset object.
        :param data: pandas DataFrame containing the dataset.
        """
        if not isinstance(data, pd.DataFrame):
            raise ValueError("The data must be a pandas DataFrame.")
        self.data = data

    def show_overview(self):
        """
        Display a general overview of the dataset.
        """
        print("Dataset Overview:")
        print(f"Number of Rows: {self.data.shape[0]}")
        print(f"Number of Columns: {self.data.shape[1]}")
        print("\nColumns:")
        print(self.data.columns.tolist())
        print("\nMissing Values:")
        print(self.data.isnull().sum())
        print("\nSample Data:")
        print(self.data.head())

    def plot_column_distribution(self, column):
        """
        Plot the distribution of values in a specific column.
        :param column: Column name to visualize.
        """
        if column not in self.data.columns:
            raise ValueError(f"Column '{column}' not found in the dataset.")
        print(f"Plotting distribution for column: {column}")
        if self.data[column].dtype == "object" or self.data[column].nunique() < 20:
            self.data[column].value_counts().plot(kind="bar", color="skyblue")
            plt.title(f"Distribution of {column}")
            plt.xlabel(column)
            plt.ylabel("Count")
            plt.show()
        else:
            sns.histplot(self.data[column], kde=True, color="skyblue")
            plt.title(f"Distribution of {column}")
            plt.xlabel(column)
            plt.ylabel("Density")
            plt.show()

    def plot_relationship(self, column1, column2):
        """
        Plot the relationship between two columns.
        :param column1: Name of the first column.
        :param column2: Name of the second column.
        """
        if column1 not in self.data.columns or column2 not in self.data.columns:
            raise ValueError(f"One or both columns '{column1}' and '{column2}' not found in the dataset.")
        print(f"Plotting relationship between '{column1}' and '{column2}'.")
        sns.scatterplot(x=self.data[column1], y=self.data[column2], alpha=0.7)
        plt.title(f"Relationship between {column1} and {column2}")
        plt.xlabel(column1)
        plt.ylabel(column2)
        plt.show()

    def plot_correlation_matrix(self):
        """
        Plot the correlation matrix for numerical features.
        """
        print("Plotting correlation matrix...")
        corr = self.data.corr()
        plt.figure(figsize=(10, 8))
        sns.heatmap(corr, annot=True, fmt=".2f", cmap="coolwarm", square=True)
        plt.title("Correlation Matrix")
        plt.show()

    def plot_histogram(self, column, bins=30, title=None):
        """
        Plot a histogram for a specified column.
        :param column: Column name to visualize.
        :param bins: Number of bins for the histogram.
        :param title: Title for the histogram.
        """
        if column not in self.data.columns:
            raise ValueError(f"Column '{column}' not found in the dataset.")
        self.data[column].hist(bins=bins, color='skyblue', edgecolor='black')
        plt.title(title or f"Histogram of {column}")
        plt.xlabel(column)
        plt.ylabel("Frequency")
        plt.show()

    def plot_grouped_histogram(self, column, group_by, bins=30):
        """
        Plot histograms of a column grouped by another column.
        :param column: Column to plot histograms for.
        :param group_by: Column to group data by.
        :param bins: Number of bins for the histogram.
        """
        if column not in self.data.columns:
            raise ValueError(f"Column '{column}' not found in the dataset.")
        if group_by not in self.data.columns:
            raise ValueError(f"Column '{group_by}' not found in the dataset.")

        unique_groups = self.data[group_by].unique()
        for group in unique_groups:
            subset = self.data[self.data[group_by] == group]
            plt.hist(subset[column], bins=bins, alpha=0.7, label=f'{group_by} = {group}')
            plt.title(f"{column} Distribution for {group_by} = {group}")
            plt.xlabel(column)
            plt.ylabel("Frequency")
            plt.legend()
            plt.show()

    def apply_function(self, func):
        """
        Apply a custom function to the dataset.
        :param func: A function that takes a DataFrame as input and returns a DataFrame.
        """
        print("Applying function to the dataset...")
        self.data = func(self.data)

    def update_dataframe(self, new_data):
        """
        Update/overwrite the existing pandas DataFrame with new data.
        :param new_data: pandas DataFrame to replace the current dataset.
        """
        if not isinstance(new_data, pd.DataFrame):
            raise ValueError("The new data must be a pandas DataFrame.")
        print("Updating the dataset...")
        self.data = new_data

    def get_dataframe(self):
        """
        Return the current pandas DataFrame.
        :return: The current DataFrame.
        """
        return self.data

