import streamlit as st
import pandas as pd
from sklearn.ensemble import StackingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split

# Load dataset
@st.cache
def load_data():
    df = pd.read_csv('C:\\Users\\meghn\\Documents\\depression_detection\\your_file.csv"')  # Replace with the actual dataset path
    return df

# Define model training function
def train_model():
    df = load_data()
    X = df.drop(columns=['depression'])  # Assuming 'depression' is the output column
    y = df['depression']

    # Split dataset
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

    # Define base models and stacking classifier
    base_models = [
        ('lr', LogisticRegression()),
        ('svc', SVC(probability=True))
    ]
    stacking_model = StackingClassifier(estimators=base_models, final_estimator=LogisticRegression())
    stacking_model.fit(X_train, y_train)
    
    return stacking_model, X.columns

# Predict depression level
def predict_depression_level(model, user_inputs):
    prediction = model.predict([user_inputs])[0]
    return int(prediction)

# Streamlit UI
def main():
    st.title("Depression Detection Quiz")
    st.write("Answer the following questions to assess your mental health.")

    # Load and train model
    model, feature_columns = train_model()

    # Dynamic question generation
    user_inputs = []
    for feature in feature_columns:
        if feature != 'depression':  # Exclude output column
            user_input = st.slider(f"{feature.capitalize()}", 0, 5, 3)
            user_inputs.append(user_input)

    if st.button("Submit"):
        depression_level = predict_depression_level(model, user_inputs)
        st.success(f"Your predicted depression level is: {depression_level}")

# Run the app
if __name__ == "__main__":
    main()
