import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.tree import DecisionTreeClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import confusion_matrix

# Load pre-trained model
model = joblib.load('best_model.joblib')
data = pd.read_csv('data.csv')

specific_columns = ['SFH', 'Request_URL', 'having_IP_Address', 'SSLfinal_State', 'web_traffic', 
                    'URL_Length', 'URL_of_Anchor', 'popUpWindow', 'Result'] 
df = data[specific_columns]

# Assuming your new dataset has features X and target variable y
X_new = df.drop(columns=['Result'])
y_new = df['Result']

# Preprocess your new data if needed
scaler = StandardScaler()
X_new_scaled = scaler.fit_transform(X_new)

# Apply the model to the new data
predictions = model.predict(X_new_scaled)

# Print predictions
print("Predictions:")
print(predictions)

# Plotting Actual vs Predicted
plt.figure(figsize=(8, 6))
plt.scatter(y_new, predictions, alpha=0.7)
plt.xlabel("Actual Values", fontsize=12)
plt.ylabel("Predicted Values", fontsize=12)
plt.title("Actual vs. Predicted", fontsize=14)
plt.grid(True, linestyle='--', linewidth=0.5)
plt.plot([0, 10], [0, 10], color='gray', linestyle='--')

# Add the confusion matrix
cm = confusion_matrix(y_new, predictions)
plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues")
plt.xlabel("Predicted labels")
plt.ylabel("True labels")
plt.title("Confusion Matrix")
plt.show()
