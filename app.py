import streamlit as st
import pandas as pd
import joblib

# Model load kar rahe hain
model = joblib.load('titanic_model.pkl')

st.title("🚢 Titanic Survival Predictor")
st.write("Is app ke zariye aap check kar sakte hain ki passenger survive karta ya nahi.")

# --- Sidebar Inputs ---
st.sidebar.header("Passenger Details")

pclass = st.sidebar.selectbox("Ticket Class", [1, 2, 3])
sex = st.sidebar.selectbox("Sex", ["male", "female"])
age = st.sidebar.slider("Age", 0, 100, 25)
sibsp = st.sidebar.number_input("Siblings/Spouses Aboard", 0, 10, 0)
parch = st.sidebar.number_input("Parents/Children Aboard", 0, 10, 0)
fare = st.sidebar.number_input("Fare Paid", 0.0, 600.0, 32.0)
embarked = st.sidebar.selectbox("Port of Embarkation", ["S", "C", "Q"])
title = st.sidebar.selectbox("Title", ["Mr", "Mrs", "Miss", "Rare"])

# --- Preprocessing Logic ---
# Data ko wahi format mein lana jo X_train mein tha
data = {
    'Pclass': [pclass],
    'Sex': [sex],             # 'Sex_male' ki jagah original
    'Age': [float(age)],
    'SibSp': [sibsp],
    'Parch': [parch],
    'Fare': [fare],
    'Embarked': [embarked],   # 'Embarked_S' ki jagah original
    'FamilySize': [sibsp + parch + 1],
    'Title': [title]          # Alag dummy columns ki jagah original
}

input_df = pd.DataFrame(data)

input_df = pd.DataFrame(data)

# --- Prediction ---
if st.button("Predict Survival"):
    try:
        prediction = model.predict(input_df)
        if prediction[0] == 1:
            st.success("Result: Survived! 🎉")
        else:
            st.error("Result: Did not survive. ❌")
    except ValueError as e:
        st.error(f"Column Mismatch Error: {e}")