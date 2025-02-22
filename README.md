🚗 Car Price Prediction Using XGBoost 🚀

Predict car prices using XGBoost with 95% accuracy! 🎯 This project demonstrates how to preprocess data, encode categorical variables, and train a powerful machine learning model to predict car prices based on features like Car Name, Fuel Type, Engine Size, and more.

Features
1. 🧹 Data Preprocessing:

Categorical features like Car Name and Fuel Type are encoded using OneHotEncoder.

Numerical features like Wheelbase and Horsepower are scaled using StandardScaler.

2. 📊 Model Training:

The XGBoost Regressor is trained from scratch every time the script runs.

3. 📈 Prediction:

Input car details, and the model predicts the price instantly.

🤖 TechStack used: 🖥️

1. XGBoost

XGBoost (Extreme Gradient Boosting) is a scalable and efficient implementation of gradient boosting, widely used for regression and classification tasks.

It is known for its high accuracy, speed, and ability to handle large datasets.

In this project, XGBoost is used as the regression model to predict car prices.

2. Scikit-learn

Scikit-learn is a powerful Python library for machine learning, providing tools for data preprocessing, model training, and evaluation.

In this project, Scikit-learn is used for:

1. Data Preprocessing: Encoding categorical variables and scaling numerical features.

2. Pipeline Creation: Combining preprocessing and model training into a single workflow.

3. Model Evaluation: Splitting data into training and testing sets.

🛠️ How It Works

1. 🧹 Data Preprocessing:

The dataset (car_data.csv) is loaded and preprocessed.

Categorical features are encoded, and numerical features are scaled.

2. 📊 Model Training:

The XGBoost Regressor is trained on the preprocessed data.

3. 📈 Prediction:

The user inputs car details, and the model predicts the price.


🖥️ How to Use:

1. Clone the repository:

2. Install dependencies:

3. Run the script:

4. Enter the car details when prompted and get the predicted price!

📂 Dataset
The dataset (car_data.csv) contains the following features:

Categorical: CarName, fueltype, aspiration, carbody, etc.

Numerical: wheelbase, carlength, curbweight, enginesize, etc.

Target Variable: price

🚀 Future Improvements
Add a Streamlit app for a user-friendly interface.

Experiment with other regression algorithms like Logistic Regression.

Deploy the model as a web application using FastAPI.

🙌 Contributions
Contributions are welcome! Feel free to open an issue or submit a pull request.

Let’s predict car prices and make smarter decisions! 🚀🚗
