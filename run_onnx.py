import onnxruntime as ort
import numpy as np
import pandas as pd

# Load the ONNX model
session = ort.InferenceSession("model.onnx")

# Get input name
input_name = session.get_inputs()[0].name
print("Input name:", input_name)

# Load your real dataset
df = pd.read_csv("asian_indian_recipes.csv")  # replace with your actual file path
print("Columns in dataset:", df.columns)  # print column names for debugging

# Separate features (X) and target (y) if applicable
# Update 'correct_column_name' to the actual column you want to drop
# Drop index column and target column (if applicable)
df = df.drop(columns=["Unnamed: 0", "cuisine", 'yeast', 'yogurt', 'zucchini'])  

# Check that the number of features is now correct (380)
print(f"Number of features: {df.shape[1]}")  # Should print 380

# If there are extra features, adjust them by selecting the first 380 columns
#if df.shape[1] > 380:
#    df = df.iloc[:, :380]  # Keep only the first 380 columns


# Convert features to numpy array
X = df.values.astype(np.float32)

# Check input shape expected by the model
print("Expected input shape:", session.get_inputs()[0].shape)

# Run predictions
predictions = session.run(None, {input_name: X})

# Print predictions
print("Predictions:", predictions[0])

# Optionally, store predictions in your dataframe and export
df['predictions'] = predictions[0]
df.to_csv("predictions_output.csv", index=False)