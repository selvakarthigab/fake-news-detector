import pandas as pd
import re
import pickle

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
model = LogisticRegression()
from sklearn.metrics import accuracy_score
from sklearn.feature_extraction.text import TfidfVectorizer
# ✅ CLEAN FUNCTION
def clean_text(text):
    if isinstance(text, str):
        text = text.lower()
        text = re.sub(r'[^a-zA-Z ]', '', text)
        return text
    else:
        return ""

# ✅ LOAD DATA
fake_df = pd.read_csv("Fake.csv", low_memory=False)
true_df = pd.read_csv("True.csv", low_memory=False)

# ✅ REMOVE EMPTY
fake_df = fake_df.dropna(subset=['text'])
true_df = true_df.dropna(subset=['text'])

# ✅ CLEAN TEXT
fake_df['text'] = fake_df['text'].apply(clean_text)
true_df['text'] = true_df['text'].apply(clean_text)

# ✅ LABELS
fake_df['label'] = 0
true_df['label'] = 1

# ✅ COMBINE
df = pd.concat([fake_df, true_df])

# ✅ INPUT / OUTPUT
X = df['text']
y = df['label']

# ✅ SPLIT
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

# ✅ TF-IDF (UPGRADE 🔥)
vectorizer = TfidfVectorizer(max_features=5000, stop_words='english', ngram_range=(1,2))
X_train_vec = vectorizer.fit_transform(X_train)
X_test_vec = vectorizer.transform(X_test)

# ✅ MODEL (BEST FOR TEXT 😎)
model = MultinomialNB()
model.fit(X_train_vec, y_train)

# ✅ TEST
y_pred = model.predict(X_test_vec)
print("Accuracy:", accuracy_score(y_test, y_pred))

# ✅ SAVE FILES
pickle.dump(model, open("model.pkl", "wb"))
pickle.dump(vectorizer, open("vectorizer.pkl", "wb"))

print("✅ Model trained and saved successfully!")