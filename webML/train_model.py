import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.ensemble import VotingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score
import joblib
import os

# 1. Load Data
# Data location: ../data/titanic/train.csv relative to this notebook
data_path = '../data/titanic/train.csv'
# Resolve absolute path for safety
data_path = os.path.abspath(os.path.join(os.path.dirname(__file__), data_path))
print(f"Reading data from: {data_path}")

df = pd.read_csv(data_path)

print("Data shape:", df.shape)

# 2. Preprocessing

# Select Features
# Numeric: Age, SibSp, Parch, Fare
# Categorical: Pclass, Sex, Embarked

numeric_features = ['Age', 'SibSp', 'Parch', 'Fare']
categorical_features = ['Pclass', 'Sex', 'Embarked']

# Preprocessing pipelines
numeric_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='median')),
    ('scaler', StandardScaler())
])

categorical_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='most_frequent')),
    ('encoder', OneHotEncoder(handle_unknown='ignore'))
])

preprocessor = ColumnTransformer(
    transformers=[
        ('num', numeric_transformer, numeric_features),
        ('cat', categorical_transformer, categorical_features)
    ])

# 3. Define Models for Voting

model1 = LogisticRegression(random_state=42, max_iter=1000)
model2 = DecisionTreeClassifier(random_state=42)
model3 = RandomForestClassifier(random_state=42)
model4 = SVC(probability=True, random_state=42) # probability=True for soft voting if needed
model5 = KNeighborsClassifier()

voting_clf = VotingClassifier(
    estimators=[
        ('lr', model1),
        ('dt', model2),
        ('rf', model3),
        ('svc', model4),
        ('knn', model5)
    ],
    voting='soft' # Use probabilities for voting
)

# Create full pipeline
full_pipeline = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('classifier', voting_clf)
])

# 4. Split Data and Train

X = df[numeric_features + categorical_features]
y = df['Survived']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train
full_pipeline.fit(X_train, y_train)

# Evaluate
y_pred = full_pipeline.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print(f"Voting Classifier Accuracy: {accuracy:.4f}")

# 5. Save Model
# Save the trained pipeline to a file to be used by the web app
joblib.dump(full_pipeline, 'voting_model.pkl')
print("Model saved to voting_model.pkl")
