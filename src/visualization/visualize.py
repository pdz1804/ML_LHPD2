"""
visualize.py

Author: Loc
Last Modified: 2025-02-25

This module provides a `DataVisualizer` class for analyzing and visualizing datasets.
It includes functions for:
- Dataset overview
- Class distribution visualization
- Feature relationships
- Word frequency and word cloud generation
- Sentiment analysis-based word distributions
- Text length analysis
- Pairwise feature relationships

"""
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import spacy
from collections import Counter
from wordcloud import WordCloud

import os
from datetime import datetime

try:
    nlp = spacy.load("en_core_web_sm")
except OSError:
    print("Mô hình 'en_core_web_sm' chưa được tải. Đang tải...")
    spacy.cli.download("en_core_web_sm")
    nlp = spacy.load("en_core_web_sm")

class DataVisualizer:
    """
    A class for visualizing and analyzing structured datasets.

    Attributes:
        data (pd.DataFrame): The dataset to be analyzed and visualized.
    """
    def __init__(self, data):
        """
        Initializes the DataVisualizer with a dataset.

        Args:
            data (pd.DataFrame): The dataset containing structured information.
        """
        if not isinstance(data, pd.DataFrame):
            raise ValueError("The data should be a pandas DataFrame.")
        self.data = data

    def plot_overview(self):
        """
        Displays an overview of the dataset, including:
        - Number of rows and columns
        - Data types and null values
        - Sample data from the dataset
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
        Plots the distribution of values in a categorical column.

        Args:
            column (str): The column name to be visualized.
        """
        if column not in self.data.columns:
            raise ValueError(f"Column '{column}' does not exist in the dataset.")
        if self.data[column].dtype == "object" or self.data[column].nunique() <= 20:
            self.data[column].value_counts().plot(kind='bar', color='skyblue')
            plt.title(f"Distribution of {column}")
            plt.xlabel(column)
            plt.ylabel("Frequency")
            save_path = f"/kaggle/working/plot_{datetime.now().strftime('%Y%m%d_%H%M%S')}.png"
            plt.savefig(save_path, dpi=300, bbox_inches="tight")
            print(f"Saved plot to {save_path}")
            plt.show()

        else:
            sns.histplot(self.data[column], kde=True, bins=30, color="skyblue")
            plt.title(f"Distribution of {column}")
            plt.xlabel(column)
            plt.ylabel("Density")
            save_path = f"/kaggle/working/plot_{datetime.now().strftime('%Y%m%d_%H%M%S')}.png"
            plt.savefig(save_path, dpi=300, bbox_inches="tight")
            print(f"Saved plot to {save_path}")
            plt.show()

    def plot_feature_relationship(self, feature1, feature2):
        """
        Plots the relationship between two numerical features using a scatter plot.

        Args:
            feature1 (str): Name of the first feature (x-axis).
            feature2 (str): Name of the second feature (y-axis).
        """
        if feature1 not in self.data.columns or feature2 not in self.data.columns:
            raise ValueError(f"One or both features '{feature1}' and '{feature2}' do not exist in the dataset.")
        plt.figure(figsize=(8, 6))
        sns.scatterplot(x=self.data[feature1], y=self.data[feature2], alpha=0.7, color="blue")
        plt.title(f"Relationship between {feature1} and {feature2}")
        plt.xlabel(feature1)
        plt.ylabel(feature2)
        save_path = f"/kaggle/working/plot_{datetime.now().strftime('%Y%m%d_%H%M%S')}.png"
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
        print(f"Saved plot to {save_path}")
        plt.show()

    def plot_all_distributions(self):
        """
        Iterates through all columns in the dataset and plots their distributions.
        Skips columns that cannot be visualized as distributions.
        """
        for column in self.data.columns:
            print(f"Plotting distribution for: {column}")
            try:
                self.plot_class_distribution(column)
            except ValueError as e:
                print(f"Skipping {column}: {e}")

    def plot_all_relationships(self):
        """
        Generates pairwise scatter plots for all numerical features in the dataset.
        If there are fewer than two numerical columns, no plots are generated.
        """
        numeric_cols = self.data.select_dtypes(include=["number"]).columns
        if len(numeric_cols) < 2:
            print("Not enough numerical features for pairwise relationships.")
            return
        sns.pairplot(self.data[numeric_cols], diag_kind="kde", plot_kws={"alpha": 0.7})
        plt.suptitle("Pairwise Relationships", y=1.02)
        save_path = f"/kaggle/working/plot_{datetime.now().strftime('%Y%m%d_%H%M%S')}.png"
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
        print(f"Saved plot to {save_path}")
        plt.show()

    def plot_class_distribution_nega_posi(self, target_column='target'):
        """
        Plots the distribution of positive and negative sentiment classes.

        Args:
            target_column (str): Column containing sentiment labels.
        """
        plt.figure(figsize=(8, 6))
        sns.countplot(x=target_column, data=self.data, palette='pastel')
        plt.title('Distribution of class Positive and class Negative (Dữ liệu mẫu)')
        plt.xlabel('(0.0 = Negative, 1.0/4.0 = Positive)')
        plt.ylabel('Ammount')
        save_path = f"/kaggle/working/plot_{datetime.now().strftime('%Y%m%d_%H%M%S')}.png"
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
        print(f"Saved plot to {save_path}")
        plt.show()

    def plot_top_words_and_wordcloud(self, text_column='text_clean', sentiment_column=None, sentiment_value=None, n_words=20, title_prefix="Toàn dataset (Dữ liệu mẫu)"):
        """
        Generates a bar chart and a word cloud for the most frequent words in a text column, optionally filtering by sentiment.

        Args:
            text_column (str): The column containing textual data.
            sentiment_column (str, optional): The column containing sentiment labels.
            sentiment_value (any, optional): The specific sentiment value to filter by.
            n_words (int): Number of top words to visualize.
            title_prefix (str): Custom title prefix for the plots.
        """
        # Filter by sentiment if specified
        if sentiment_column and sentiment_value is not None:
            filtered_data = self.data[self.data[sentiment_column] == sentiment_value]
            if filtered_data.empty:
                print(f"No data found for sentiment = {sentiment_value}")
                return
            all_text = ' '.join(filtered_data[text_column].dropna().astype(str))
            title_prefix = f"{title_prefix} - Sentiment {sentiment_value}"
        else:
            all_text = ' '.join(self.data[text_column].dropna().astype(str))

        # Increase spaCy's max_length if text is too large
        nlp.max_length = len(all_text) + 1000

        words = [token.text for token in nlp(all_text.lower()) if token.is_alpha] 
        words = [word for word in words if word.isalnum()]
        
        word_counts = Counter(words)
        top_words = word_counts.most_common(n_words)
        
        # Bar Plot
        plt.figure(figsize=(10, 6))
        df_words = pd.DataFrame({'Word': [w for w, _ in top_words], 'Frequency': [c for _, c in top_words]})
        sns.barplot(data=df_words, x='Frequency', y='Word', palette='Blues_d')
        plt.title(f'{title_prefix} - Top {n_words} Words')
        plt.xlabel('Frequency')
        plt.ylabel('Word')

        save_path = f"/kaggle/working/top_words_{title_prefix.replace(' ', '_')}.png"
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
        print(f"Saved bar plot to {save_path}")

        plt.show()
        
        # Word Cloud
        wordcloud = WordCloud(width=800, height=400, max_words=n_words, background_color='white', colormap='viridis').generate(all_text)
        plt.figure(figsize=(10, 6))
        plt.imshow(wordcloud, interpolation='bilinear')
        plt.axis('off')
        plt.title(f'Word Cloud {title_prefix}')

        save_path = f"/kaggle/working/wordcloud_{title_prefix.replace(' ', '_')}.png"
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
        print(f"Saved word cloud to {save_path}")

        plt.show()

    def plot_class_distribution_nega_posi(self, target_column='target'):
        """
        Plot the distribution of positive and negative classes.
        """
        plt.figure(figsize=(8, 6))
        sns.countplot(x=target_column, data=self.data, palette='pastel')
        plt.title('Distribution of class Positive and class Negative (Dữ liệu mẫu)')
        plt.xlabel('(0.0 = Negative, 1.0/4.0 = Positive)')
        plt.ylabel('Ammount')
        plt.show()

    def plot_positive_words_and_wordcloud(self, text_column='text_clean', n_words=20):
        """
        Plots the most frequent words and a word cloud for positive sentiment samples.

        Args:
            text_column (str): The column containing textual data.
            n_words (int): Number of top words to visualize.
        """
        positive_data = self.data[self.data['target'] == 4.0] 
        if len(positive_data) == 0:
            print("There is no positive in the dataset.")
            return
        self.plot_top_words_and_wordcloud(text_column=text_column, sentiment_column="target", sentiment_value=4, n_words=n_words, title_prefix="Class Positive (data_sample)")

    def plot_negative_words_and_wordcloud(self, text_column='text_clean', n_words=20):
        """
        Plots the most frequent words and a word cloud for negative sentiment samples.

        Args:
            text_column (str): The column containing textual data.
            n_words (int): Number of top words to visualize.
        """
        negative_data = self.data[self.data['target'] == 0.0]
        if len(negative_data) == 0:
            print("There is no negative in the dataset.")
            return
        self.plot_top_words_and_wordcloud(text_column=text_column, sentiment_column="target", sentiment_value=0, n_words=n_words, title_prefix="Class Negative (data_sample)")
        
    def plot_text_length_distribution(self, target_column='target', length_columns=['text_length', 'text_clean_length']):
        """
        Plots the distribution of text lengths across sentiment categories.

        Args:
            target_column (str): Column containing sentiment labels.
            length_columns (list): List of columns representing text lengths.
        
        Returns:
            list: A list of matplotlib figures containing box plots and histograms.
        """
        figs = []
        for length_col in length_columns:
            # Box plot
            plt.figure(figsize=(10, 6))
            sns.boxplot(x=target_column, y=length_col, data=self.data, hue=target_column, palette='pastel')
            plt.title(f'Distribution of {length_col} by sentiment')
            plt.xlabel('Sentiment (0.0 = Negative, 1.0/4.0 = Positive)')
            plt.ylabel(f'Length ({length_col})')
            figs.append(plt.gca())
            plt.close() 

            # Histogram
            plt.figure(figsize=(10, 6))
            sns.histplot(data=self.data, x=length_col, hue=target_column, kde=True, bins=30, palette='pastel')
            plt.title(f'Histogram of {length_col} by sentiment')
            plt.xlabel(f'Length ({length_col})')
            plt.ylabel('Frequency')
            figs.append(plt.gca())
            plt.close()  
        return figs[:2]  
        
    def plot_word_frequency_by_sentiment(self, text_column='text_clean', target_column='target', n_words=10):
        """
        Compares word frequency between positive and negative sentiment samples.

        Args:
            text_column (str): The column containing textual data.
            target_column (str): Column containing sentiment labels.
            n_words (int): Number of top words to compare.
        """
        unique_targets = self.data[target_column].unique()
        print(f"Unique values in {target_column}: {unique_targets}")

        positive_value = 4.0
        negative_value = 0.0

        self.data[target_column] = self.data[target_column].astype(float).dropna()

        positive_data = self.data[self.data[target_column] == positive_value]
        negative_data = self.data[self.data[target_column] == negative_value]

        if len(positive_data) == 0 or len(negative_data) == 0:
            print("Not enough positive or negative data to plot.")
            return

        positive_text = ' '.join(positive_data[text_column].dropna().astype(str))
        positive_words = [token.text for token in nlp(positive_text.lower()) if token.is_alpha and not token.is_stop]
        positive_counts = Counter(positive_words).most_common(n_words)

        negative_text = ' '.join(negative_data[text_column].dropna().astype(str))
        negative_words = [token.text for token in nlp(negative_text.lower()) if token.is_alpha and not token.is_stop]
        negative_counts = Counter(negative_words).most_common(n_words)

        all_words = set([word for word, _ in positive_counts] + [word for word, _ in negative_counts])
        words = sorted(list(all_words))
        positive_counts_dict = dict(positive_counts)
        negative_counts_dict = dict(negative_counts)

        freq_positive = [positive_counts_dict.get(word, 0) for word in words]
        freq_negative = [negative_counts_dict.get(word, 0) for word in words]


        if len(words) != len(freq_positive) or len(words) != len(freq_negative):
            raise ValueError("Length mismatch in word frequency lists")

        df_freq = pd.DataFrame({
            'Word': words,
            'Positive': freq_positive,
            'Negative': freq_negative
        })

        plt.figure(figsize=(12, 8))
        pivot_table = df_freq.pivot_table(index='Word', values=['Positive', 'Negative']).fillna(0)
        sns.heatmap(pivot_table, annot=True, fmt='.0f', cmap='YlOrRd')
        plt.title('Word Frequency by Sentiment (Positive vs Negative)')
        plt.ylabel('Word')
        save_path = f"/kaggle/working/plot_{datetime.now().strftime('%Y%m%d_%H%M%S')}.png"
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
        print(f"Saved plot to {save_path}")
        plt.show()

        df_freq_long = df_freq.melt(id_vars=['Word'], var_name='Sentiment', value_name='Frequency')
        plt.figure(figsize=(12, 6))
        sns.barplot(data=df_freq_long, x='Word', y='Frequency', hue='Sentiment', palette='pastel')
        plt.title('Comparison of Word Frequency between Positive and Negative')
        plt.xlabel('Word')
        plt.ylabel('Frequency')
        plt.xticks(rotation=45)
        save_path = f"/kaggle/working/plot_{datetime.now().strftime('%Y%m%d_%H%M%S')}.png"
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
        print(f"Saved plot to {save_path}")
        plt.show()

        
    def plot_length_sentiment_scatter(self, length_column='text_length', target_column='target', size_column='text_clean_length'):
        """
        Plots a scatter plot showing the relationship between text length and sentiment.

        Args:
            length_column (str): The column representing text length.
            target_column (str): Column containing sentiment labels.
            size_column (str): Column representing the size of cleaned text.
        """
        plt.figure(figsize=(12, 8))
        scatter = sns.scatterplot(data=self.data, x=length_column, y=target_column, 
                                hue=target_column, size=size_column, sizes=(20, 200), 
                                alpha=0.6, palette='pastel')
        plt.title(f'Relationship between {length_column} length and sentiment')
        plt.xlabel(f'Length ({length_column})')
        plt.ylabel('Sentiment (0.0 = Negative, 1.0/4.0 = Positive)')
        plt.legend(title='Sentiment', bbox_to_anchor=(1.05, 1), loc='upper left')
        plt.tight_layout()
        save_path = f"/kaggle/working/plot_{datetime.now().strftime('%Y%m%d_%H%M%S')}.png"
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
        print(f"Saved plot to {save_path}")
        plt.show()

