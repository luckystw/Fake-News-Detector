import os
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
import pickle

# Updated dataset URL and path
DATASET_URL = 'https://raw.githubusercontent.com/GeorgeMcIntire/fake_real_news_dataset/main/fake_and_real_news_dataset.csv'
DATASET_PATH = 'fake_and_real_news_dataset.csv'

if not os.path.exists(DATASET_PATH):
    import requests
    print('Downloading dataset...')
    r = requests.get(DATASET_URL)
    with open(DATASET_PATH, 'wb') as f:
        f.write(r.content)

# Load dataset
print('Loading dataset...')
df = pd.read_csv(DATASET_PATH)
if 'text' not in df.columns or 'label' not in df.columns:
    raise ValueError('Dataset must have "text" and "label" columns.')

X = df['text']
y = df['label'].apply(lambda x: 0 if x == 'FAKE' else 1)  # 0: Fake, 1: Real

# Split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Vectorize
vectorizer = TfidfVectorizer(stop_words='english', max_features=5000)
X_train_vec = vectorizer.fit_transform(X_train)
X_test_vec = vectorizer.transform(X_test)

# Models
models = {
    'logistic_regression': LogisticRegression(max_iter=1000),
    'decision_tree': DecisionTreeClassifier(),
    'gradient_boost': GradientBoostingClassifier(),
    'random_forest': RandomForestClassifier()
}

# Train and save
os.makedirs('models', exist_ok=True)
for name, model in models.items():
    print(f'Training {name}...')
    model.fit(X_train_vec, y_train)
    with open(f'models/{name}.pkl', 'wb') as f:
        pickle.dump(model, f)

# Save vectorizer
with open('models/vectorizer.pkl', 'wb') as f:
    pickle.dump(vectorizer, f)

print('All models trained and saved in models/ directory.') 