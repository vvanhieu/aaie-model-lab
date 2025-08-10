# streamlit_app.py
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import torch
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
)
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
from rouge_score import rouge_scorer
from bert_score import score as bert_score_fn

#1. EvaluateModel

## i. Dataset Structure Instruction
### a. Classification Model Evaluation
dataset = {

    "prompts": ["", "", "", ...], #List of the text,
    "labels": ["AI", "Human", "neutral", ...],  # Ground-truth labels (can be int or str)
    "predictions": ["positive", "positive", "neutral", ...]  # Predicted labels (same format as `label`)
}


### b. Generative Model Evaluation

dataset = {
    "reference_texts": [
        ["The cat sat on the mat.", "A cat is sitting on a mat."],
        ["Hello, how are you?"]
        # Each item is a list of reference texts
    ],
    "generated_texts": [
        "The cat is sitting on the mat.",
        "Hi, how do you do?"
        # Each item is a generated hypothesis
    ]
}


# ==================================
# CLASS DEFINITION
# ==================================
class EvaluateModel:
    def __init__(self, dataset=None, model=None, tokenizer=None):
        self.model = model
        self.tokenizer = tokenizer
        self.results = {}
        self.device = ("cuda" if torch.cuda.is_available()
                       else "mps" if torch.backends.mps.is_available()
                       else "cpu")
        self.dataset = dataset

    def encode_labels(self, col):
        try:
            int_labels = [int(x) for x in self.dataset[col]]
            label_names = sorted(set(int_labels))
        except ValueError:
            le = LabelEncoder()
            encoded_labels = le.fit_transform(self.dataset[col])
            int_labels = encoded_labels
            label_names = sorted(set(encoded_labels))
        return int_labels, label_names

    def bleu_score(self, reference_texts, generated_text):
        ref_tokens = [ref.split() for ref in reference_texts]
        gen_tokens = generated_text.split()
        smoothie = SmoothingFunction().method1
        bleu = sentence_bleu(ref_tokens, gen_tokens, smoothing_function=smoothie)
        return {'BLEU': bleu}

    def rouge_score(self, reference_texts, generated_text):
        scorer = rouge_scorer.RougeScorer(['rouge1', 'rouge2', 'rougeL'], use_stemmer=True)
        scores = [scorer.score(ref, generated_text) for ref in reference_texts]
        avg_scores = {k.upper(): np.mean([s[k].fmeasure for s in scores]) for k in scores[0]}
        return avg_scores

    def bert_score(self, reference_texts, generated_text, model_type='bert-base-uncased'):
        P, R, F1 = bert_score_fn([generated_text], [reference_texts], model_type=model_type)
        return {
            'BERTScore_Precision': P.mean().item(),
            'BERTScore_Recall': R.mean().item(),
            'BERTScore_F1': F1.mean().item()
        }

    def evaluate_classification_model(self, average='macro'):
        targets, label_names_target = self.encode_labels("labels")
        predictions, _ = self.encode_labels("predictions")
        acc = accuracy_score(targets, predictions)
        prec = precision_score(targets, predictions, average=average, zero_division=0)
        rec = recall_score(targets, predictions, average=average, zero_division=0)
        f1 = f1_score(targets, predictions, average=average, zero_division=0)
        cm = confusion_matrix(targets, predictions, labels=label_names_target)

        self.results = {
            "Accuracy": acc,
            "Precision": prec,
            "Recall": rec,
            "F1-score": f1
        }

        fig, ax = plt.subplots()
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                    xticklabels=label_names_target, yticklabels=label_names_target, ax=ax)
        ax.set_xlabel("Predicted")
        ax.set_ylabel("Actual")
        ax.set_title("Confusion Matrix")
        return fig, self.results

    def evaluate_generative_model(self, metrics=['bleu', 'rouge', 'bertscore'], bert_model='bert-base-uncased'):
        refs = self.dataset['reference_texts']
        gens = self.dataset['generated_texts']
        all_scores = {}

        for ref, gen in zip(refs, gens):
            if 'bleu' in metrics:
                all_scores.update(self.bleu_score([ref], gen))
            if 'rouge' in metrics:
                all_scores.update(self.rouge_score([ref], gen))
            if 'bertscore' in metrics:
                all_scores.update(self.bert_score(ref, gen, bert_model))
        self.results = all_scores
        return self.results

# STREAMLIT UI

# Create padding columns so interface is centered
col_pad_left, col_main, col_pad_right = st.columns([1, 2, 1])

with col_main:
    st.set_page_config(page_title="Model Evaluation Dashboard", layout="wide")
    st.title("Model Evaluation Tool")   
    mode = st.radio("What type of model do you want to test with?", ["Classification", "Generative"])
    uploaded_file = st.file_uploader("Upload dataset (CSV)", type=['csv'])

    if mode == "Classification":
        avg_type = st.selectbox("Average type", ["macro", "micro", "weighted"])
    else:
        selected_metrics = st.multiselect("Select metrics", ["bleu", "rouge", "bertscore"], default=["bleu", "rouge"])

    submit_button = st.button("Run Evaluation")

    # Placeholders for output
    graph_placeholder = st.empty()
    dataset_preview = st.empty()
    results_placeholder = st.empty()

# Processing after button click
if submit_button and uploaded_file:
    df = pd.read_csv(uploaded_file)
    dataset_preview.write(df.head())
    evaluator = EvaluateModel(dataset=df)

    if mode == "Classification":
        fig, results = evaluator.evaluate_classification_model(average=avg_type)
        graph_placeholder.pyplot(fig)
        results_placeholder.write(pd.DataFrame([results]))

    elif mode == "Generative":
        results = evaluator.evaluate_generative_model(metrics=selected_metrics)
        results_placeholder.write(pd.DataFrame([results]))
