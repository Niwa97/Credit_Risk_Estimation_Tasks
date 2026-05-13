import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    RocCurveDisplay, roc_curve, roc_auc_score,
    ConfusionMatrixDisplay, confusion_matrix
)
from xgboost import XGBClassifier

data = pd.read_csv('DataPD.csv')
data_regression = data.replace(99.0, np.nan)

#Separation of independent and dependednt variable
X = data_regression.drop(columns=['ID', 'deflag'])
Y = data_regression['deflag']

#Split train/test sets
X_train, X_test, Y_train, Y_test = train_test_split(
    X, Y, stratify=Y, random_state=285, train_size=0.8
)

print(f'X_train: {X_train.shape}, Y_train: {Y_train.shape}')
print(f'X_test: {X_test.shape}, Y_test: {Y_test.shape}')
print(f'Default ratio - train: {100 * Y_train.mean():.4f}%')
print(f'Default ratio - test:  {100 * Y_test.mean():.4f}%')

xgb_model = XGBClassifier(
    n_estimators=100,
    learning_rate=0.05,
    max_depth=4,
    random_state=285,
    eval_metric='auc'
)
xgb_model.fit(X_train, Y_train)


# ROC Curve
Y_score = xgb_model.predict_proba(X_test)[:, 1]
fp_rate, tp_rate, _ = roc_curve(Y_test, Y_score)
roc_display = RocCurveDisplay(fpr=fp_rate, tpr=tp_rate)
roc_display.plot()
plt.title("XGBoost ROC Curve")
plt.show()

# AUC Score
auc = roc_auc_score(Y_test, Y_score)
print(f"XGBoost AUC score: {auc:.4f}")

# Confusion Matrix
Y_pred = xgb_model.predict(X_test)
cm = confusion_matrix(Y_test, Y_pred)
ConfusionMatrixDisplay(cm).plot(cmap='Blues')
plt.title("XGBoost Confusion Matrix")
plt.show()


for bank_id, name in [(484, 'ABC'), (47, 'XYZ'), (2741, 'QQQ')]:
    row = data_regression[data_regression['ID'] == bank_id].drop(columns=['deflag', 'ID'])
    if not row.empty:
        pd_value = xgb_model.predict_proba(row)[0, 1]
        print(f'PD_{name} (ID: {bank_id}) = {pd_value * 100:.2f}%')
    else:
        print(f'ID {bank_id} not found in dataset.')
