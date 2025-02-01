import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report

def train_model(features_path, output_model_path):
    """
    Train a logistic regression model on extracted features.
    """
    features = pd.read_csv(features_path)
    X = features
    y = pd.read_csv("data/processed/processed_data.csv")['target']

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    model = LogisticRegression()
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    print(classification_report(y_test, y_pred))
    print(f"Model saved to {output_model_path}")

if __name__ == "__main__":
    features_path = os.path.join("data", "processed", "features.csv")
    output_model_path = os.path.join("models", "trained", "model.pkl")
    train_model(features_path, output_model_path)
