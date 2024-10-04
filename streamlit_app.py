import pandas as pd
import streamlit as st
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
from sklearn.impute import SimpleImputer

# Load your dataset
data_path = 'dataset_02052023.csv'  # Change this to your dataset path
df = pd.read_csv(data_path)

# Display the dataset in the Streamlit app
st.write("Dataset Preview:")
st.dataframe(df.head())

# Check for initial shape and missing values
st.write("Initial Dataset Info:")
st.write(df.info())
st.write("Missing Values Count per Column:")
st.write(df.isnull().sum())

# Convert datetime columns to numeric timestamps
for column in df.select_dtypes(include=['object']):
    try:
        df[column] = pd.to_datetime(df[column], errors='coerce')
        df[column] = df[column].astype('int64') // 10**9
    except Exception as e:
        st.warning(f"Could not convert column '{column}' to datetime: {e}")

# Check and display the updated DataFrame
st.write("Updated Dataset Preview:")
st.dataframe(df.head())

# Select the target column
target_column = st.selectbox("Select the target column:", df.columns.tolist())

if target_column in df.columns:
    X = df.drop(columns=[target_column])
    y = df[target_column]
else:
    st.error(f"Column '{target_column}' not found in the dataset!")
    st.stop()

# Convert y to numeric (if necessary) and handle non-numeric values
y = pd.to_numeric(y, errors='coerce')

# Fill NaN values in y with the mean (for regression purposes)
y = y.fillna(y.mean())

# Impute missing values in X with column mean using SimpleImputer
imputer = SimpleImputer(strategy='mean')
X = imputer.fit_transform(X)

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Initialize the model
model = LinearRegression()

# Train the model
model.fit(X_train, y_train)

# Make predictions on the test set
y_pred = model.predict(X_test)

# Calculate and display the mean squared error
mse = mean_squared_error(y_test, y_pred)
st.write(f"Mean Squared Error: {mse}")

# Optionally, display the first few predictions
st.write("Sample Predictions:")
st.write(pd.DataFrame({'Actual': y_test[:5].values, 'Predicted': y_pred[:5]}))
