import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    classification_report,
    roc_auc_score,
    roc_curve,
)
import matplotlib.pyplot as plt

synthetic_data = {
    'CGPA': np.random.uniform(6.0, 10.0, size=200),
    'GRE_Score': np.random.randint(290, 340, size=200),
    'GMAT_Score': np.random.randint(400, 800, size=200),
    'TOEFL_Score': np.random.randint(80, 120, size=200),
    'Published_Conferences': np.random.randint(0, 3, size=200),
    'Published_Journals': np.random.randint(0, 3, size=200),
    'Mini_Projects': np.random.randint(0, 3, size=200),
    'Internships': np.random.randint(0, 3, size=200),
    'Admission_Status': np.random.choice([0, 1], size=200, p=[0.4, 0.6]),
}

admissions_df = pd.DataFrame(synthetic_data)

admissions_df['Total_Publications'] = (
    admissions_df['Published_Conferences'] + admissions_df['Published_Journals']
)

X_features = admissions_df.drop(['Admission_Status', 'Total_Publications'], axis=1)
y_target = admissions_df['Admission_Status']

X_train, X_test, y_train, y_test = train_test_split(
    X_features, y_target, test_size=0.2, random_state=42, stratify=y_target
)

logistic_model = LogisticRegression(max_iter=1000)
logistic_model.fit(X_train, y_train)

predicted_labels = logistic_model.predict(X_test)
predicted_probs = logistic_model.predict_proba(X_test)[:, 1]

model_accuracy = accuracy_score(y_test, predicted_labels)
confusion_mat = confusion_matrix(y_test, predicted_labels)
classification_metrics = classification_report(y_test, predicted_labels)
roc_auc_value = roc_auc_score(y_test, predicted_probs)

print(f"Model Accuracy: {model_accuracy:.2f}\n")
print("Confusion Matrix:\n", confusion_mat)
print("\nClassification Report:\n", classification_metrics)
print(f"ROC AUC Score: {roc_auc_value:.2f}\n")

fpr, tpr, roc_thresholds = roc_curve(y_test, predicted_probs)
plt.figure(figsize=(10, 6))
plt.plot(fpr, tpr, label=f'Logistic Regression (AUC = {roc_auc_value:.2f})', color='blue')
plt.plot([0, 1], [0, 1], linestyle='--', color='gray')
plt.title('ROC Curve - Logistic Regression')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.legend(loc='lower right')
plt.grid(alpha=0.5)
plt.show()