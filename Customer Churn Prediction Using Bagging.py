import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import BaggingClassifier
from sklearn.metrics import accuracy_score

X, y = make_classification(n_samples=1000, n_features=5, n_informative=3, n_redundant=0, random_state=42)

df = pd.DataFrame(X, columns=['Age', 'Monthly_Bill', 'Total_Usage', 'Customer_Service_Calls', 'Contract_Length'])
df['Churn'] = y

print("--- Dataset Summary ---")
print(df.head())
print(df.describe())

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

tree_model = DecisionTreeClassifier(random_state=42)
tree_model.fit(X_train, y_train)
y_pred_tree = tree_model.predict(X_test)
tree_acc = accuracy_score(y_test, y_pred_tree)

bagging_model = BaggingClassifier(estimator=DecisionTreeClassifier(), n_estimators=50, random_state=42)
bagging_model.fit(X_train, y_train)
y_pred_bagging = bagging_model.predict(X_test)
bag_acc = accuracy_score(y_test, y_pred_bagging)

print("\n--- Model Comparison ---")
print(f"Single Decision Tree Accuracy: {tree_acc * 100:.2f}%")
print(f"Bagging Classifier Accuracy:   {bag_acc * 100:.2f}%")

models = ['Decision Tree', 'Bagging']
accuracies = [tree_acc, bag_acc]

plt.bar(models, accuracies, color=['blue', 'green'])
plt.ylabel('Accuracy')
plt.title('Decision Tree vs Bagging Accuracy')
plt.ylim(0, 1)
plt.show()