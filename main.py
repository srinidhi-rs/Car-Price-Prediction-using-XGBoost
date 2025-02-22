import pandas as pd
from xgboost import XGBRegressor
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.model_selection import train_test_split

# Load the dataset
df = pd.read_csv('car_data.csv')  # Replace with your dataset file name

# Define features (X) and target (y)
X = df.drop('price', axis=1)  # Replace 'price' with your target column name
y = df['price']

# Define categorical and numerical features
categorical_features = ['CarName', 'fueltype', 'aspiration', 'doornumber', 'carbody',
                        'drivewheel', 'enginelocation', 'enginetype', 'cylindernumber', 'fuelsystem']
numerical_features = ['wheelbase', 'carlength', 'carwidth', 'carheight', 'curbweight',
                      'enginesize', 'boreratio', 'stroke', 'compressionratio', 'horsepower',
                      'peakrpm', 'citympg', 'highwaympg']

# Preprocessing pipeline
preprocessor = ColumnTransformer(
    transformers=[
        ('num', StandardScaler(), numerical_features),  # Scale numerical features
        ('cat', OneHotEncoder(handle_unknown='ignore'), categorical_features)  # Encode categorical features
    ])

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Create the XGBoost Regressor pipeline
model = Pipeline(steps=[
    ('preprocessor', preprocessor),  # Preprocess the data
    ('regressor', XGBRegressor(random_state=42))  # XGBoost Regressor
])

# Train the model
print("Training the XGBoost model...")
model.fit(X_train, y_train)
print("XGBoost model trained successfully!")


def get_user_input():
    # Collect user input for all features
    print("Enter the following details to predict car price:")
    CarName = input("Car Name: ")
    fueltype = input("Fuel Type: ")
    aspiration = input("Aspiration: ")
    doornumber = input("Door Number: ")
    carbody = input("Car Body: ")
    drivewheel = input("Drive Wheel: ")
    enginelocation = input("Engine Location: ")
    wheelbase = float(input("Wheelbase: "))
    carlength = float(input("Car Length: "))
    carwidth = float(input("Car Width: "))
    carheight = float(input("Car Height: "))
    curbweight = int(input("Curb Weight: "))
    enginetype = input("Engine Type: ")
    cylindernumber = input("Cylinder Number: ")
    enginesize = int(input("Engine Size: "))
    fuelsystem = input("Fuel System: ")
    boreratio = float(input("Bore Ratio: "))
    stroke = float(input("Stroke: "))
    compressionratio = float(input("Compression Ratio: "))
    horsepower = int(input("Horsepower: "))
    peakrpm = int(input("Peak RPM: "))
    citympg = int(input("City MPG: "))
    highwaympg = int(input("Highway MPG: "))

    # Create a DataFrame from the input data
    query = pd.DataFrame([[CarName, fueltype, aspiration, doornumber, carbody, drivewheel, enginelocation, wheelbase,
                           carlength, carwidth, carheight, curbweight, enginetype, cylindernumber, enginesize,
                           fuelsystem, boreratio, stroke, compressionratio, horsepower, peakrpm, citympg, highwaympg]],
                         columns=['CarName', 'fueltype', 'aspiration', 'doornumber', 'carbody', 'drivewheel',
                                  'enginelocation', 'wheelbase', 'carlength', 'carwidth', 'carheight', 'curbweight',
                                  'enginetype', 'cylindernumber', 'enginesize', 'fuelsystem', 'boreratio', 'stroke',
                                  'compressionratio', 'horsepower', 'peakrpm', 'citympg', 'highwaympg'])

    return query


def predict_car_price():
    # Get user input
    query = get_user_input()

    # Predict car price
    prediction = model.predict(query)[0]
    print(f"\nPredicted Car Price: ${prediction:.2f}")


# Run the prediction function
if __name__ == '__main__':
    predict_car_price()