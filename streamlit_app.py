import streamlit as st
import pandas as pd
from sklearn.ensemble import StackingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from sklearn.exceptions import NotFittedError

# Load dataset
@st.cache_data
def load_data(file_path=None, uploaded_file=None):
    if uploaded_file is not None:
        df = pd.read_csv(uploaded_file)
    elif file_path is not None:
        df = pd.read_csv(file_path)
    else:
        raise FileNotFoundError("No valid file path or uploaded file provided.")
    return df

# Define model training function
def train_model(df):
    X = df.drop(columns=['depression'])  # Assuming 'depression' is the output column
    y = df['depression']

    # Split dataset
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

    # Define base models and stacking classifier
    base_models = [
        ('lr', LogisticRegression(max_iter=1000)),
        ('svc', SVC(probability=True))
    ]
    stacking_model = StackingClassifier(estimators=base_models, final_estimator=LogisticRegression(max_iter=1000))
    stacking_model.fit(X_train, y_train)
    
    return stacking_model, X.columns

# Predict depression level
def predict_depression_level(model, user_inputs):
    try:
        prediction = model.predict([user_inputs])[0]
        return int(prediction)
    except NotFittedError:
        st.error("The model has not been trained yet.")
        return None

# Streamlit UI
def main():
    st.title("Depression Detection Quiz")
    st.write("Answer the following questions to assess your mental health.")

    # File upload option
    uploaded_file = st.file_uploader("your_file.csv")
    file_path = 'C:\\Users\\meghn\\Documents\\depression_detection\\your_file.csv'
    if uploaded_file or file_path:
        try:
            df = load_data(file_path=file_path, uploaded_file=uploaded_file)
            st.write("Dataset loaded successfully!")
            st.write(df.head())

            # Load and train model
            model, feature_columns = train_model(df)

            # Dynamic question generation
            user_inputs = []
            for feature in feature_columns:
                user_input = st.slider(f"{feature.capitalize()}", 0, 5, 3)
                user_inputs.append(user_input)

            if st.button("Submit"):
                depression_level = predict_depression_level(model, user_inputs)
                if depression_level is not None:
                    st.success(f"Your predicted depression level is: {depression_level}")
        except Exception as e:
            st.error(f"An error occurred: {e}")
    else:
        st.warning("Please upload a dataset to proceed.")

# Run the app
if __name__ == "__main__":
    main()
