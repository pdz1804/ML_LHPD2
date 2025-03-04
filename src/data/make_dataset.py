"""
make_dataset.py

A module for defining a Dataset class to visualize and explore datasets.

Author: Nguyen Quang Phu
Last Modified: 2025-01-21

This module includes:
- A Dataset class for dataset visualization and exploration.
- Methods for displaying dataset overviews, plotting distributions, relationships, and correlation matrices.
- Methods for applying custom functions and updating the dataset.
"""

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

class Dataset:
    """
    A class for dataset visualization and exploration.

    Attributes:
        data (pd.DataFrame): The dataset to be visualized and explored.
    """

    def __init__(self, data):
        """
        Initialize the Dataset object.

        Args:
            data (pd.DataFrame): The dataset to be visualized and explored.

        Raises:
            ValueError: If the data is not a pandas DataFrame.
        """
        if not isinstance(data, pd.DataFrame):
            raise ValueError("The data must be a pandas DataFrame.")
        self.data = data

    def show_overview(self):
        """
        Display a general overview of the dataset.
        """
        try:
            print("Dataset Overview:")
            print(f"Number of Rows: {self.data.shape[0]}")
            print(f"Number of Columns: {self.data.shape[1]}")
            print("\nColumns:")
            print(self.data.columns.tolist())
            print("\nMissing Values:")
            print(self.data.isnull().sum())
            print("\nSample Data:")
            print(self.data.head())
        except Exception as e:
            print(f"An error occurred while showing the overview: {e}")

    def plot_column_distribution(self, column):
        """
        Plot the distribution of values in a specific column.

        Args:
            column (str): The column name to visualize.

        Raises:
            ValueError: If the column is not found in the dataset.
        """
        try:
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
        except Exception as e:
            print(f"An error occurred while plotting column distribution: {e}")

    def plot_relationship(self, column1, column2):
        """
        Plot the relationship between two columns.

        Args:
            column1 (str): The name of the first column.
            column2 (str): The name of the second column.

        Raises:
            ValueError: If one or both columns are not found in the dataset.
        """
        try:
            if column1 not in self.data.columns or column2 not in self.data.columns:
                raise ValueError(f"One or both columns '{column1}' and '{column2}' not found in the dataset.")
            print(f"Plotting relationship between '{column1}' and '{column2}'.")
            sns.scatterplot(x=self.data[column1], y=self.data[column2], alpha=0.7)
            plt.title(f"Relationship between {column1} and {column2}")
            plt.xlabel(column1)
            plt.ylabel(column2)
            plt.show()
        except Exception as e:
            print(f"An error occurred while plotting relationship: {e}")

    def plot_correlation_matrix(self):
        """
        Plot the correlation matrix for numerical features.
        """
        try:
            print("Plotting correlation matrix...")
            corr = self.data.corr()
            plt.figure(figsize=(10, 8))
            sns.heatmap(corr, annot=True, fmt=".2f", cmap="coolwarm", square=True)
            plt.title("Correlation Matrix")
            plt.show()
        except Exception as e:
            print(f"An error occurred while plotting correlation matrix: {e}")

    def plot_histogram(self, column, bins=30, title=None):
        """
        Plot a histogram for a specified column.

        Args:
            column (str): The column name to visualize.
            bins (int): The number of bins for the histogram.
            title (str, optional): The title for the histogram. Defaults to None.

        Raises:
            ValueError: If the column is not found in the dataset.
        """
        try:
            if column not in self.data.columns:
                raise ValueError(f"Column '{column}' not found in the dataset.")
            self.data[column].hist(bins=bins, color='skyblue', edgecolor='black')
            plt.title(title or f"Histogram of {column}")
            plt.xlabel(column)
            plt.ylabel("Frequency")
            plt.show()
        except Exception as e:
            print(f"An error occurred while plotting histogram: {e}")

    def plot_grouped_histogram(self, column, group_by, bins=30):
        """
        Plot histograms of a column grouped by another column.

        Args:
            column (str): The column to plot histograms for.
            group_by (str): The column to group data by.
            bins (int): The number of bins for the histogram.

        Raises:
            ValueError: If the column or group_by column is not found in the dataset.
        """
        try:
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
        except Exception as e:
            print(f"An error occurred while plotting grouped histogram: {e}")

    def apply_custom_function(self, func, *args, **kwargs):
        """
        Apply a custom function to the dataset.

        This method allows users to apply any custom function to the dataset. The custom function
        can take the dataset as input and return a modified dataset.

        Args:
            func (function): A custom function that takes the dataset as input and returns a modified dataset.
            *args: Additional positional arguments to pass to the custom function.
            **kwargs: Additional keyword arguments to pass to the custom function.

        Returns:
            pd.DataFrame: The modified dataset after applying the custom function.
        """
        try:
            print("Applying custom function to the dataset...")
            self.data = func(self.data, *args, **kwargs)
            return self.data
        except Exception as e:
            print(f"An error occurred while applying custom function: {e}")

    def add_new_column(self, column_name, func, *args, **kwargs):
        """
        Add a new column to the dataset by applying a custom function to each row.

        This method allows users to add a new column to the dataset by applying a custom function to each row.
        The custom function can take a row as input and return a value for the new column.

        Args:
            column_name (str): The name of the new column to be added.
            func (function): A custom function that takes a row as input and returns a value for the new column.
            *args: Additional positional arguments to pass to the custom function.
            **kwargs: Additional keyword arguments to pass to the custom function.

        Returns:
            pd.DataFrame: The dataset with the new column added.
        """
        try:
            print(f"Adding new column '{column_name}' to the dataset...")
            self.data[column_name] = self.data.apply(lambda row: func(row, *args, **kwargs), axis=1)
            return self.data
        except Exception as e:
            print(f"An error occurred while adding new column: {e}")

    def filter_rows(self, condition_func, *args, **kwargs):
        """
        Filter rows in the dataset based on a custom condition function.

        This method allows users to filter rows in the dataset based on a custom condition function.
        The custom function can take a row as input and return a boolean value indicating whether the row should be included.

        Args:
            condition_func (function): A custom function that takes a row as input and returns a boolean value.
            *args: Additional positional arguments to pass to the custom function.
            **kwargs: Additional keyword arguments to pass to the custom function.

        Returns:
            pd.DataFrame: The filtered dataset.
        """
        try:
            print("Filtering rows in the dataset...")
            self.data = self.data[self.data.apply(lambda row: condition_func(row, *args, **kwargs), axis=1)]
            return self.data
        except Exception as e:
            print(f"An error occurred while filtering rows: {e}")
    

    def update_dataframe(self, new_data):
        """
        Update/overwrite the existing pandas DataFrame with new data.

        Args:
            new_data (pd.DataFrame): The new data to replace the current dataset.

        Raises:
            ValueError: If the new data is not a pandas DataFrame.
        """
        try:
            if not isinstance(new_data, pd.DataFrame):
                raise ValueError("The new data must be a pandas DataFrame.")
            print("Updating the dataset...")
            self.data = new_data
        except Exception as e:
            print(f"An error occurred while updating the dataframe: {e}")

    def get_dataframe(self):
        """
        Return the current pandas DataFrame.

        Returns:
            pd.DataFrame: The current dataset.
        """
        try:
            return self.data
        except Exception as e:
            print(f"An error occurred while getting the dataframe: {e}")

    def get_numeric_columns(self):
        """
        Return a list of column names that have numeric values.

        Returns:
            list: A list of column names with numeric values.
        """
        try:
            numeric_columns = self.data.select_dtypes(include=['number']).columns.tolist()
            return numeric_columns
        except Exception as e:
            print(f"An error occurred while getting numeric columns: {e}")

    def get_columns_by_type(self, dtype):
        """
        Return a list of column names that have the specified data type.

        Args:
            dtype (str): The data type to filter columns by (e.g., 'number', 'object').

        Returns:
            list: A list of column names with the specified data type.
        """
        try:
            columns_by_type = self.data.select_dtypes(include=[dtype]).columns.tolist()
            return columns_by_type
        except Exception as e:
            print(f"An error occurred while getting columns by type: {e}")

# Example usage:
# if __name__ == "__main__":
#     import pandas as pd
#     from make_dataset import Dataset
#
#     # Load a sample dataset
#     df = pd.read_csv("path/to/your/dataset.csv")
#
#     # Initialize the Dataset object
#     dataset = Dataset(df)
#
#     # Show an overview of the dataset
#     dataset.show_overview()
#
#     # Plot the distribution of a specific column
#     dataset.plot_column_distribution("column_name")
#
#     # Plot the relationship between two columns
#     dataset.plot_relationship("column1", "column2")
#
#     # Plot the correlation matrix
#     dataset.plot_correlation_matrix()
#
#     # Apply a custom function to the dataset
#     def custom_function(data):
#         # Perform some custom operations on the data
#         return data
#
#     dataset.apply_function(custom_function)
#
#     # Update the dataset with new data
#     new_df = pd.read_csv("path/to/new_dataset.csv")
#     dataset.update_dataframe(new_df)
#
#     # Get the current dataset
#     current_df = dataset.get_dataframe()

