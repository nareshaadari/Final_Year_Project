import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, mean_squared_error
import joblib

# Load your dataset
df = pd.read_csv('modif_eluru_weather_data_with_category.csv')

# Convert 'date' column to datetime
df['date'] = pd.to_datetime(df['date'], format='%d-%m-%Y')

# Create new features from 'date'
df['month'] = df['date'].dt.month
df['day'] = df['date'].dt.day
df['dayofyear'] = df['date'].dt.dayofyear

# Separating features and target variable
X = df[['month', 'day', 'dayofyear']]

# Label Encoding for the target variable
label_encoder = LabelEncoder()
df['weather_encoded'] = label_encoder.fit_transform(df['weather'])
y = df['weather_encoded']

# Save the label encoder for future use
joblib.dump(label_encoder, 'label_encoder.pkl')

# Scaling features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Save the scaler for future use
joblib.dump(scaler, 'scaler.pkl')

# Split the data into train and test sets
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

# Initialize and train RandomForestClassifier
best_model = RandomForestClassifier()
best_model.fit(X_train, y_train)

# Evaluate RandomForestClassifier
predictions = best_model.predict(X_test)
accuracy = accuracy_score(y_test, predictions)
mse = mean_squared_error(y_test, predictions)

# Save the best model to a file
joblib.dump(best_model, 'model.pkl')