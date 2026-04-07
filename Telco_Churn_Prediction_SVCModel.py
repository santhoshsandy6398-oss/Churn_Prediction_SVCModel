from sklearn.svm import SVC
from sklearn.metrics import precision_score, recall_score, f1_score, confusion_matrix, ConfusionMatrixDisplay
import matplotlib.pyplot as plt

# Initialize and train the best model (SVM Classifier)
best_model = SVC(kernel='linear', random_state=42)
best_model.fit(X_train, y_train)

# Make predictions
y_pred_train_best = best_model.predict(X_train)
y_pred_test_best = best_model.predict(X_test)

# Calculate and print metrics for the best model
print("SVM Classifier Performance:")

precision_train_best = precision_score(y_train, y_pred_train_best, average='weighted', zero_division=0)
recall_train_best = recall_score(y_train, y_pred_train_best, average='weighted', zero_division=0)
f1_train_best = f1_score(y_train, y_pred_train_best, average='weighted', zero_division=0)

precision_test_best = precision_score(y_test, y_pred_test_best, average='weighted', zero_division=0)
recall_test_best = recall_score(y_test, y_pred_test_best, average='weighted', zero_division=0)
f1_test_best = f1_score(y_test, y_pred_test_best, average='weighted', zero_division=0)

print(f"  Precision (Train): {precision_train_best:.4f}")
print(f"  Recall (Train): {recall_train_best:.4f}")
print(f"  F1-score (Train): {f1_train_best:.4f}")
print(f"  Precision (Test): {precision_test_best:.4f}")
print(f"  Recall (Test): {recall_test_best:.4f}")
print(f"  F1-score (Test): {f1_test_best:.4f}")

# Display Confusion Matrix for the Test Set
cm = confusion_matrix(y_test, y_pred_test_best)
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=best_model.classes_)
disp.plot(cmap=plt.cm.Blues)
plt.title('Confusion Matrix for SVM Classifier (Test Set)')
plt.show()
