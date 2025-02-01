import os
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer

def build_features(processed_data_path, features_save_path):
    """
    Create TF-IDF features from text data.
    """
    data = pd.read_csv(processed_data_path)
    vectorizer = TfidfVectorizer(max_features=1000)
    tfidf_features = vectorizer.fit_transform(data['text'])
    features_df = pd.DataFrame(tfidf_features.toarray(), columns=vectorizer.get_feature_names_out())
    print(f"Saving features to {features_save_path}")
    features_df.to_csv(features_save_path, index=False)

if __name__ == "__main__":
    processed_data_path = os.path.join("data", "processed", "processed_data.csv")
    features_save_path = os.path.join("data", "processed", "features.csv")
    build_features(processed_data_path, features_save_path)
