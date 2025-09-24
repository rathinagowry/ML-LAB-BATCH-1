
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report,roc_auc_score
from sklearn.pipeline import Pipeline
import numpy as np

df=pd.read_csv('heart.csv')
df.head()
RANDOM_STATE=42

X=df.drop('target',axis=1)
y=df['target']

X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.2,random_state=42,stratify=y)

pipelines = {
    'L1': Pipeline([
        ('scaler', StandardScaler()),
        ('logreg', LogisticRegression(penalty='l1', solver='saga', max_iter=5000, random_state=RANDOM_STATE))
    ]),
    'L2': Pipeline([
        ('scaler', StandardScaler()),
        ('logreg', LogisticRegression(penalty='l2', solver='lbfgs', max_iter=5000, random_state=RANDOM_STATE))
    ]),
    'ElasticNet': Pipeline([
        ('scaler', StandardScaler()),
        ('logreg', LogisticRegression(penalty='elasticnet', solver='saga', l1_ratio=0.5, max_iter=5000, random_state=RANDOM_STATE))
    ])
}

# Train, predict, and evaluate each model
results = {}

for reg_type, pipeline in pipelines.items():
    print(f"\nTraining Logistic Regression with {reg_type} regularization...")
    pipeline.fit(X_train, y_train)

    y_pred = pipeline.predict(X_test)
    y_proba = pipeline.predict_proba(X_test)[:, 1]

    acc = accuracy_score(y_test, y_pred)
    roc_auc = roc_auc_score(y_test, y_proba)
    report = classification_report(y_test, y_pred)

    results[reg_type] = {
        'accuracy': acc,
        'roc_auc': roc_auc,
        'classification_report': report
    }

    print(f"Accuracy: {acc:.4f}")
    print(f"ROC AUC: {roc_auc:.4f}")
    print("Classification Report:")
    print(report)