import gradio as gr
import joblib
import pandas as pd
import numpy as np

# Load the trained model
# Ensure the model file exists in the same directory or provide the correct path
try:
    model = joblib.load('voting_model.pkl')
    print("Model loaded successfully.")
except FileNotFoundError:
    print("Error: 'voting_model.pkl' not found. Please train the model using 'myModel.ipynb' first.")
    model = None

def predict_survival(pclass, sex, age, sibsp, parch, fare, embarked):
    if model is None:
        return "Model not loaded. Please train the model first."
    
    # Create a DataFrame from inputs
    data = pd.DataFrame({
        'Pclass': [int(pclass)],
        'Sex': [sex],
        'Age': [age],
        'SibSp': [int(sibsp)],
        'Parch': [int(parch)],
        'Fare': [fare],
        'Embarked': [embarked]
    })
    
    # Predict probabilities
    try:
        proba = model.predict_proba(data)[0]
        prediction = model.predict(data)[0]
        
        result = "Survived" if prediction == 1 else "Deceased"
        probability = f"{max(proba) * 100:.2f}%"
        
        return f"Prediction: {result} (Confidence: {probability})"
    except Exception as e:
        return f"Error during prediction: {str(e)}"

# Define the Gradio interface
iface = gr.Interface(
    fn=predict_survival,
    inputs=[
        gr.Radio([1, 2, 3], label="Pclass (Ticket Class)", info="1 = 1st, 2 = 2nd, 3 = 3rd"),
        gr.Radio(["male", "female"], label="Sex"),
        gr.Slider(0, 100, value=30, label="Age"),
        gr.Number(value=0, label="SibSp (Siblings/Spouses aboard)"),
        gr.Number(value=0, label="Parch (Parents/Children aboard)"),
        gr.Number(value=32.2, label="Fare"),
        gr.Radio(["C", "Q", "S"], label="Embarked", info="C = Cherbourg, Q = Queenstown, S = Southampton")
    ],
    outputs="text",
    title="Titanic Survivor Prediction",
    description="Enter passenger details to predict survival probability."
)

if __name__ == "__main__":
    iface.launch(server_name="0.0.0.0")
