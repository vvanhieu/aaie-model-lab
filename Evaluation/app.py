# streamlit_app.py
import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from evaluate_model import EvaluateModel, process_model  # Assuming evaluate_model.py is in the same directory

# --- Streamlit Page Config ---
st.set_page_config(page_title="Model Evaluation Dashboard", layout="wide")

# --- Page Layout ---
col_main, col_pad_right = st.columns(2)

with col_main:
    st.title("Model Evaluation Tool")
    
    # --- Mode Selection ---
    mode = st.selectbox("🔍 What type of model do you want to test with?", ["Classification", "Generative"])
    data_mode = st.selectbox("📂 What type of data are you using?", ["Simple Text", "CSV"])
    
    # --- Data Input ---
    if data_mode == "CSV":
        data = st.file_uploader("Upload dataset (CSV)", type=['csv'])
    elif data_mode == "Simple Text" and mode == "Classification":
        data = st.text_area("Enter your text data (one entry per line):")
        label = st.text_input("Enter labels for the text data (comma-separated):")
    elif data_mode == "Simple Text" and mode == "Generative":
        data = st.text_area("Enter your reference and generated texts (format: ref1|gen1, ref2|gen2):")
    
    # --- Few-Shot Prompt ---
    prompt = st.text_area("Enter few-shot prompt (optional):", height=150)

    # --- Options ---
    if mode == "Classification":
        avg_type = st.selectbox("Average type", ["Macro", "Micro", "Weighted", "Binary"])
    else:
        selected_metrics = st.multiselect("📏 Select metrics", ["Bleu", "Rouge", "Bertscore"], default=["Bleu", "Rouge"])

    # --- Buttons ---
    col_btn1, col_btn2, col_btn3 = st.columns(3)
    with col_btn1:
        process_button = st.button("Process Model")
    with col_btn2:
        submit_button = st.button("Run Evaluation")
    with col_btn3:
        process_eval_button = st.button("Process & Evaluate")
with col_pad_right:
    # --- Placeholders ---
    dataset_preview = st.empty()
    results_container = st.container()
    matrix_container = st.container()


# --- Session State for DataFrame ---
if "df" not in st.session_state:
    st.session_state.df = pd.DataFrame()


# --- Processing Logic ---
def run_processing():
    if data_mode == "CSV":
        st.session_state.df = pd.read_csv(data)
        dataset_preview.write(st.session_state.df.head())
    elif data_mode == "Simple Text":
        if data.strip():
            text_data = data.splitlines()
            st.session_state.df = pd.DataFrame({"prompts": text_data, "labels": label})
            if prompt:
                st.session_state.df = process_model(st.session_state.df, few_shot_prompt=prompt)
            else:
                st.session_state.df = process_model(st.session_state.df)
        else:
            st.error("Please enter some text data.")


# --- Evaluation Logic ---
def run_evaluation():
    if st.session_state.df.empty:
        st.error("No processed data found. Please process your data first.")
        return
    
    evaluator = EvaluateModel(dataset=st.session_state.df)
    
    if mode == "Classification":
        evaluator.evaluate_classification_model(average=avg_type.lower())
        
        # Show metrics
        results_container.write("### 📊 Evaluation Results")
        results_container.write(pd.DataFrame([evaluator.results]))
        
        # Show confusion matrix
        if "labels" in st.session_state.df.columns and "predictions" in st.session_state.df.columns:
            labels = st.session_state.df["labels"]
            preds = st.session_state.df["predictions"]
            
            fig, ax = plt.subplots()
            cm = pd.crosstab(labels, preds)
            sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", ax=ax)
            ax.set_xlabel("Predictions")
            ax.set_ylabel("True Labels")
            ax.set_title("Confusion Matrix")
            matrix_container.pyplot(fig)
        else:
            matrix_container.warning("No 'labels' and 'predictions' columns found for confusion matrix.")
    
    elif mode == "Generative":
        results = evaluator.evaluate_generative_model(metrics=selected_metrics)
        results_containers.write(pd.DataFrame([results]))


# --- Button Actions ---
if process_button:
    run_processing()

if submit_button:
    run_evaluation()

if process_eval_button:
    run_processing()
    run_evaluation()
