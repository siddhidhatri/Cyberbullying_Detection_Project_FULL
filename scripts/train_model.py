from preprocessing import preprocess_data, vectorize_text
from sklearn.naive_bayes import MultinomialNB
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
import joblib
import os

def train():
    print("🔄 Loading and preprocessing data...")
    df = preprocess_data("data/cyberbullying_tweets.csv")
    X, vectorizer = vectorize_text(df["clean_text"])
    y = df["cyberbullying_type"]

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    print("⚙️ Training Multinomial Naive Bayes model...")
    model = MultinomialNB()
    model.fit(X_train, y_train)

    os.makedirs("models", exist_ok=True)
    joblib.dump(model, "models/multiclass_model.pkl")
    joblib.dump(vectorizer, "models/vectorizer.pkl")

    print("✅ Model trained and saved to /models")
    print("\n📊 Classification Report:")
    y_pred = model.predict(X_test)
    print(classification_report(y_test, y_pred))
