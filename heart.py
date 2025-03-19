import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf

import pandas as pd

df = pd.read_csv("HeartPredictonQuantuDataset.csv")  # Replace with your actual file path

data = df.values

x = np.array(data[:, :-1])
y = data[:, -1]


loaded_model = tf.keras.models.load_model("my_model.keras")


m, n = x.shape

fig, axes = plt.subplots(10,5, figsize=(10,5))
fig.tight_layout(pad=0.1,rect=[0, 0.03, 1, 0.92]) #[left, bottom, right, top]

for i,ax in enumerate(axes.flat):
    # Select random indices
    random_index = np.random.randint(m)
    
    arr = np.array(x[random_index])
    row_arr = arr.reshape(1, -1)

    pred = np.round(loaded_model.predict(row_arr)),
    
    ax.set_title(f"{y[random_index]} , {pred[0][0][0]} = {y[random_index] == pred[0][0][0]}")
    ax.set_axis_off()
    
    
fig.suptitle('Age,Gender,BloodPressure,Cholesterol,HeartRate,QuantumPatternFeature', fontsize=16)
plt.show()


# Age = int(input("Enter Age: "))
# Gender = int(input("Enter Gender: "))
# BloodPressure = int(input("Enter BloodPressure: "))
# Cholesterol = int(input("Enter Cholesterol: "))
# HeartRate = int(input("Enter HeartRate: "))
# QuantumPatternFeature = float(input("Enter QuantumPatternFeature: "))

# prediction = loaded_model.predict(np.array([[
#     Age,
#     Gender,
#     BloodPressure,
#     Cholesterol,
#     HeartRate,
#     QuantumPatternFeature
# ]]))

# print(f"Predicted result: {np.round(prediction[0][0])}")



# fig, ax = plt.subplots(figsize=(5, 5))
# ax.set_axis_off()

# result = "Heart Disease" if np.round(prediction[0][0]) == 1 else "Healthy"
# ax.set_title(f"Predicted result: {result}", fontsize=16)
# fig.tight_layout(pad=0.1, rect=[0, 0.03, 1, 0.92])  # [left, bottom, right, top]

# fig.suptitle('Prediction Results', fontsize=16)
# plt.show()
