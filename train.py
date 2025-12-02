# train.py : ตัวอย่าง ML ง่ายที่สุด
# pip install scikit-learn
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
import joblib

# 1) Load data
data = load_iris()
X_train, X_test, y_train, y_test = train_test_split(data.data, data.target, test_size=0.2)

# 2) Train model
model = RandomForestClassifier()
model.fit(X_train, y_train)

# 3) Save model + accuracy
acc = model.score(X_test, y_test)
joblib.dump(model, "model.pkl")

with open("result.txt", "w") as f:
    f.write(f"Accuracy = {acc}")

print("Training completed! Accuracy =", acc)
