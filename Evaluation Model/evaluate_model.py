import nltk
import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
from rouge_score import rouge_scorer
from transformers import AutoTokenizer, AutoModelForCausalLM
from bert_score import score as bert_score_fn
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import (accuracy_score, precision_score, recall_score, f1_score,  confusion_matrix, classification_report)
from transformers import GPT2LMHeadModel, GPT2TokenizerFast, Trainer, TrainingArguments
from transformers import TextDataset, DataCollatorForLanguageModeling
import streamlit as st
"""#1. EvaluateModel

## i. Dataset Structure Instruction
### a. Classification Model Evaluation
dataset = {

    "prompts": ["", "", "", ...], #List of the text,
    "labels": ["AI", "Human", "neutral", ...],  # Ground-truth labels (can be int or str)
    "predictions": ["positive", "positive", "neutral", ...]  # Predicted labels (same format as `label`)
}

We can evaluate the model as
```
evaluator = EvaluateModel(dataset=dataset)
evaluator.evaluate_classification_model(average='macro',print_result)
```
The function already support label encoding automatically if your labels are strings.


### b. Generative Model Evaluation

```
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
```
We can evaluate the model as


```
evaluator = EvaluateModel(dataset=dataset)
evaluator.evaluate_generative_model(metrics=['bleu', 'rouge', 'bertscore'], print_result=True) # Head to the functio to see more detail
```
"""

class EvaluateModel():
    def __init__(self, dataset=None, model=None, tokenizer=None) -> None:
      self.model = model
      self.tokenizer = tokenizer
      self.results = {}
      self.device = ( "cuda" if torch.cuda.is_available()
                    else "mps" if torch.backends.mps.is_available()
                    else "cpu")

      self.dataset = dataset

    #Check the labels is int and encode it if it is string
    def encode_labels(self, col):
        try:
            int_labels = [int(x) for x in self.dataset[col]]
            label_names = sorted(set(int_labels))
        except ValueError:
            le = LabelEncoder()
            encoded_labels = le.fit_transform(self.dataset[col])
            int_labels = encoded_labels
            label_names = sorted(set(encoded_labels))  # integer labels only

            print(f"The {col} labels have been encoded. Integer mapping:")
            for original, encoded in zip(le.classes_, le.transform(le.classes_)):
                print(f"{original} -> {encoded}")

        return int_labels, label_names

    #Compute the BLEU score
    def bleu_score(self, reference_texts, generated_text):
        ref_tokens = [ref.split() for ref in reference_texts]
        gen_tokens = generated_text.split()
        smoothie = SmoothingFunction().method1
        bleu = sentence_bleu(ref_tokens, gen_tokens, smoothing_function=smoothie)
        return {'bleu': bleu}

    #Compute the ROUGE score
    def rouge_score(self, reference_texts, generated_text):
        scorer = rouge_scorer.RougeScorer(['rouge1', 'rouge2', 'rougeL'], use_stemmer=True)
        scores = [scorer.score(ref, generated_text) for ref in reference_texts]
        avg_scores = {
            k: np.mean([s[k].fmeasure for s in scores]) for k in scores[0]
        }
        return avg_scores

    #Compute the BERTScore
    def bert_score(self, reference_texts, generated_text, model_type_ = 'bert-base-uncased'):
        P, R, F1 = bert_score_fn([generated_text], [reference_texts], model_type=model_type_)
        return {
            'bertscore_precision': P.mean().item(),
            'bertscore_recall': R.mean().item(),
            'bertscore_f1': F1.mean().item()
        }

    #Evaluate the clasification model
    def evaluate_classification_model(self, average='macro', print_result=False):
        """
        Evaluate a classification model using standard metrics:
        - Accuracy, Precision, Recall, F1-score, Confusion Matrix

        Args:
            average (str): Averaging method for multi-class ('binary', 'micro', 'macro').
            print_result (bool): Whether to print results and plot confusion matrix.

        Workflow:
            - Checks for 'label' and 'prediction' keys in dataset.
            - Encodes non-integer labels if necessary.
            - Computes classification metrics.
            - Optionally prints a classification report and confusion matrix.
        """
        if any(k not in self.dataset for k in ["labels", "predictions"]):
            print(self.dataset)
            print("Please provide the dataset with 'label' and 'prediction' columns.")
            return
        # Encode labels (handles string → int mapping if needed)
        targets, label_names_target = self.encode_labels("labels")
        predictions, label_names_prediction = self.encode_labels("predictions")

        label_names = label_names_target

        # Compute evaluation metrics
        acc = accuracy_score(targets, predictions)
        prec = precision_score(targets, predictions, average=average, zero_division=0)
        rec = recall_score(targets, predictions, average=average, zero_division=0)
        f1 = f1_score(targets, predictions, average=average, zero_division=0)
        cm = confusion_matrix(targets, predictions, labels=label_names)

        # Optionally print results
        if print_result:
            plt.figure(figsize=(6, 5))
            sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=label_names, yticklabels=label_names)
            plt.xlabel('Predicted')
            plt.ylabel('Actual')
            plt.title('Confusion Matrix')
            plt.show()

            print(f"Accuracy: {acc:.4f}")
            print(f"F1-score: {f1:.4f}")
            print(f"Recall: {rec:.4f}")
            print(f"Precision: {prec:.4f}")
            print("\nClassification Report:")
            print(classification_report(targets, predictions, labels=label_names, zero_division=0))

        # Store results
        self.results.update({
            "accuracy": acc,
            "precision": prec,
            "recall": rec,
            "f1_score": f1,
        })

    def evaluate_generative_model(self, metrics=['bleu', 'rouge', 'bertscore'], print_result = True, bert_model='bert-base-uncased'):
        """
        Evaluate a generative model using metrics like BLEU, ROUGE, and BERTScore.

        Args:
            metrics (list): List of metrics to compute. Options: 'bleu', 'rouge', 'bertscore'.
            bert_model (str): Hugging Face model name to use for BERTScore.

        Workflow:
            - Validates that 'reference_texts' and 'generated_texts' are in the dataset.
            - Loops over each pair of reference and generated text.
            - Computes and stores selected metrics.
        """
        if "reference_texts" not in self.dataset or "generated_texts" not in self.dataset:
            print("Please provide 'reference_texts' and 'generated_texts' in the dataset.")
            return

        refs = self.dataset['reference_texts'].copy()
        gens = self.dataset['generated_texts'].copy()

        for ref, gen in zip(refs, gens):
            # Expect `ref` to be a list of strings, `gen` to be a single string
            if 'bleu' in metrics:
                self.results.update(self.bleu_score(ref, gen))
            if 'rouge' in metrics:
                self.results.update(self.rouge_score(ref, gen))
            if 'bertscore' in metrics:
                self.results.update(self.bert_score(ref, gen, bert_model))

        if print_result:
            print("Generative Model Evaluation Results:")
            for metric, score in self.results.items():
                print(f"{metric:20}: {score:.4f}")


def classify_text(model, tokenizer, device, text, few_shot_prompt, max_new_tokens=10):
    # Insert the new text into the few-shot prompt
    prompt = few_shot_prompt.format(text)

    # Tokenize and send to device
    input_ids = tokenizer(prompt, return_tensors="pt").to(device)
    output_ids = model.generate(**input_ids, max_new_tokens=max_new_tokens)

    generated_text = tokenizer.decode(output_ids[0], skip_special_tokens=True)
    # Extract the label predicted after the last 'Label:' token
    # For example, output may be the full prompt + " AI" or " Human"
    label = generated_text.split("Label:")[-1].strip().split()[0]

    return label

default_prompt = """
Decide whether the following text was written by a human or an AI.

Text: "Artificial intelligence is a powerful tool for automating tasks."
Label: AI

Text: "I walked to the market this morning and bought fresh bread."
Label: Human

Text: "The moon is a celestial body that orbits Earth."
Label: AI

Text: "Yesterday, I enjoyed a long walk in the park."
Label: Human

Text: "Machine learning models improve with more data."
Label: AI

Text: "{}"
Label: Human or AI
"""

def process_model(data, few_shot_prompt=default_prompt):
    print("Processing model with few-shot prompt...")
    device = "cuda" if torch.cuda.is_available() else "mps"
    model = AutoModelForCausalLM.from_pretrained(
        "./phi-2", 
        torch_dtype=torch.float16, 
        device_map="auto"
    )
    tokenizer = AutoTokenizer.from_pretrained("./phi-2")
    pred_label_ls = []

    # Create UI placeholders
    progress_bar = st.progress(0)
    log_area = st.empty()  # For showing logs
    logs = ""  # Store logs as text

    total = len(data["prompts"])
    for i, text in enumerate(data["prompts"]):
        pred_label = classify_text(model, tokenizer, device, text, few_shot_prompt)
        pred_label_ls.append(pred_label)

        # Update logs
        logs += f"Text: {text}\nPredicted Label: {pred_label} | Actual Label: {data['labels'][i]}\n\n"
        log_area.text(logs)

        # Update progress bar
        progress_bar.progress((i + 1) / total)

    # Store the prediction to dataset
    data['predictions'] = pred_label_ls
    return data