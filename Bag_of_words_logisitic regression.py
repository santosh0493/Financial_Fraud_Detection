import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from sklearn.ensemble import RandomForestClassifier  # Using RandomForest instead of XGBoost
from sklearn.metrics import classification_report, accuracy_score
import re
import warnings
warnings.filterwarnings('ignore')

# Download NLTK resources explicitly
import nltk
try:
    nltk.download('stopwords', quiet=True)
    from nltk.corpus import stopwords
    stop_words = set(stopwords.words('english'))
    nltk_available = True
except Exception as e:
    print(f"Warning: Could not download NLTK stopwords: {e}")
    print("Continuing without stopword removal...")
    nltk_available = False
    stop_words = set()  # Empty set as fallback

# Function to preprocess text
def preprocess_text(text):
    # Convert to lowercase
    text = text.lower()
    # Remove special characters and digits
    text = re.sub(r'[^\w\s]', ' ', text)
    text = re.sub(r'\d+', ' ', text)
    # Remove extra whitespace
    text = re.sub(r'\s+', ' ', text).strip()
    # Remove stopwords if available
    if nltk_available:
        text = ' '.join([word for word in text.split() if word not in stop_words])
    return text

# Load the data
# Assuming your CSV has columns 'Fillings' (text) and 'Fraud' (yes/no)
# df = pd.read_csv('your_financial_data.csv')
df = pd.read_csv('/Users/santosh0493/Documents/EE964-Project/Final_Dataset.csv')
# For demonstration, I'll create a sample dataframe
# df = pd.DataFrame({
#     'Fillings': ['nanitem 14 exhibits financial statements reports form 10k index exhibits following documents...'],
#     'Fraud': ['No']
# })

# # Add more dummy examples for demonstration purposes
# df = pd.concat([df] * 5, ignore_index=True)
# df.iloc[1:3, 1] = 'Yes'  # Mark some as fraud

# Preprocess the text
df['clean_text'] = df['Fillings'].apply(preprocess_text)

# Split the data
X_train, X_test, y_train, y_test = train_test_split(
    df['clean_text'],
    (df['Fraud'].str.lower() == 'yes').astype(int),  # Convert to binary, case-insensitive
    test_size=0.2,
    random_state=42
)

print(f"Training set size: {len(X_train)}")
print(f"Test set size: {len(X_test)}")

# Approach 1: Bag-of-Words + Logistic Regression
print("\n--- Approach 1: Bag-of-Words + Logistic Regression ---")
# Create bag of words features
bow_vectorizer = CountVectorizer(max_features=5000)
X_train_bow = bow_vectorizer.fit_transform(X_train)
X_test_bow = bow_vectorizer.transform(X_test)

# Train logistic regression model
lr_model = LogisticRegression(max_iter=1000, class_weight='balanced')
lr_model.fit(X_train_bow, y_train)

# Evaluate
y_pred_lr = lr_model.predict(X_test_bow)
print(f"Accuracy: {accuracy_score(y_test, y_pred_lr):.4f}")
print(classification_report(y_test, y_pred_lr))

# Approach 2: TF-IDF + Linear SVM
print("\n--- Approach 2: TF-IDF + Linear SVM ---")
# Create TF-IDF features
tfidf_vectorizer = TfidfVectorizer(max_features=5000)
X_train_tfidf = tfidf_vectorizer.fit_transform(X_train)
X_test_tfidf = tfidf_vectorizer.transform(X_test)

# Train SVM model
svm_model = LinearSVC(class_weight='balanced', max_iter=10000)
svm_model.fit(X_train_tfidf, y_train)

# Evaluate
y_pred_svm = svm_model.predict(X_test_tfidf)
print(f"Accuracy: {accuracy_score(y_test, y_pred_svm):.4f}")
print(classification_report(y_test, y_pred_svm))

# Approach 3: RandomForest on TF-IDF (replacing XGBoost + word embeddings)
print("\n--- Approach 3: RandomForest on TF-IDF (alternative to XGBoost) ---")
# Using the TF-IDF features from approach 2
rf_model = RandomForestClassifier(n_estimators=100, class_weight='balanced', random_state=42)
rf_model.fit(X_train_tfidf, y_train)

# Evaluate
y_pred_rf = rf_model.predict(X_test_tfidf)
print(f"Accuracy: {accuracy_score(y_test, y_pred_rf):.4f}")
print(classification_report(y_test, y_pred_rf))

# Compare the three approaches
print("\n--- Comparison of the three approaches ---")
print(f"Bag-of-Words + Logistic Regression accuracy: {accuracy_score(y_test, y_pred_lr):.4f}")
print(f"TF-IDF + Linear SVM accuracy: {accuracy_score(y_test, y_pred_svm):.4f}")
print(f"RandomForest on TF-IDF accuracy: {accuracy_score(y_test, y_pred_rf):.4f}")