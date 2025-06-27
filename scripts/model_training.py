from preprocessing import preprocess_data, vectorize_text
from sklearn.naive_bayes import MultinomialNB
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
import joblib
import os

# Load and preprocess data
df = preprocess_data("data/cyberbullying_tweets.csv")
X, vectorizer = vectorize_text(df["clean_text"])
y = df["cyberbullying_type"]

# Split data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train model
model = MultinomialNB()
model.fit(X_train, y_train)

# Save model and vectorizer
os.makedirs("models", exist_ok=True)
joblib.dump(model, "models/multiclass_model.pkl")
joblib.dump(vectorizer, "models/vectorizer.pkl")

# Evaluate
y_pred = model.predict(X_test)
print(classification_report(y_test, y_pred))
