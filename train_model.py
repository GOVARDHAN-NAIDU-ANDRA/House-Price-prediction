import pandas as pd
import pickle
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeRegressor
from sklearn.preprocessing import LabelEncoder

# Load dataset
data = pd.read_csv('train.csv')

# Encode categorical values (Location)
le = LabelEncoder()
data['Location_Encoded'] = le.fit_transform(data['Location'])

# Define features & target
features = ['Area', 'Location_Encoded', 'No. of Bedrooms', 'New/Resale',
            'Gymnasium', 'Car Parking', 'Indoor Games', 'Jogging Track']
X = data[features]
y = data['Price'] / 1e6  # Normalize price for better prediction

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train Decision Tree model
model = DecisionTreeRegressor()
model.fit(X_train, y_train)

# Save model & encoder
pickle.dump(model, open('model.pkl', 'wb'))
pickle.dump(le, open('label_encoder.pkl', 'wb'))

print("✅ Model and Label Encoder saved successfully!")
