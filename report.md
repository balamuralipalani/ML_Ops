# MLOps Assignment 2 
ML OPS assignment for Hugging Face Fine-Tuning, Experiment Tracking &amp; Model Deployment

**Name:** Balamurali Palaniyandi

**Roll Number:** G25AIT2070  

**Program:** PGD AI

**Date:** 25 May 2026

---

## 1. Introduction & Objective

This project implements the MLOps Assignment 2 pipeline by fine-tuning a pre-trained **DistilBERT** model on **UCSD Goodreads reviews** to classify books into **eight genres**. The work demonstrates end-to-end MLOps practices: production scripting, experiment tracking, cloud GPU training, evaluation with artifact logging, and model deployment.

**Core MLOps practices demonstrated:**
- **Productionizing notebooks:** Modular Python scripts for data loading, training, evaluation, and deployment.
- **Experiment tracking:** Logging metrics, hyperparameters, and artifacts via **Weights & Biases (W&B)**.
- **Cloud GPU training:** Leveraging **Kaggle's** free GPU infrastructure.
- **Model deployment:** Publishing the fine-tuned model to the **Hugging Face Hub**.
- **Reproducibility:** Scripts and tracked experiments ensure the workflow is repeatable.

---

## 2. Dataset

| Attribute | Value |
|-----------|-------|
| **Source** | [UCSD Book Graph](https://mengtingwan.github.io/data/goodreads.html) |
| **Genres** | poetry, children, comics_graphic, fantasy_paranormal, history_biography, mystery_thriller_crime, romance, young_adult |
| **Features** | Review text only |
| **Size** | 8 genres × 2,000 reviews sampled = ~16,000 total |
| **Preprocessing** | Tokenized with DistilBERT tokenizer; truncated/padded to 512 tokens |
| **Split** | 800 training / 200 test per genre (80/20 split) |

The dataset is loaded via HTTP streaming from UCSD mirrors, sampled to 2,000 reviews per genre for computational efficiency, and split stratified by genre.

---

## 3. Model & Training Setup

**Model:** `distilbert-base-cased`  
**Tokenizer:** `DistilBertTokenizerFast`  
**Classes:** 8 genres  
**Max length:** 512 tokens  
**Device:** CUDA (Kaggle T4 x2)

**Hyperparameters:**

| Parameter | Value |
|-----------|-------|
| Epochs | 3 |
| Batch size (train) | 10 |
| Batch size (eval) | 16 |
| Learning rate | 5e-5 |
| Warmup steps | 100 |
| Weight decay | 0.01 |
| Evaluation strategy | steps |
| Logging | every 100 steps |

DistilBERT (66M parameters) was selected because it is lighter and faster than full BERT while retaining strong performance, making it ideal for educational fine-tuning on limited GPU time.

---

## 4. Implementation Highlights

### 4.1 Kaggle Secrets Management
API tokens are stored securely via **Kaggle Secrets** rather than hard-coding:
- `WANDB_API_KEY` — W&B authentication.
- `HF_TOKEN` — Hugging Face Hub write access.

```python
from kaggle_secrets import UserSecretsClient
secrets = UserSecretsClient()
WANDB_API_KEY = secrets.get_secret('WANDB_API_KEY')
HF_TOKEN = secrets.get_secret('HF_TOKEN')
```

### 4.2 Weights & Biases Integration
W&B is initialized before training with the full hyperparameter configuration:

```python
wandb.init(
    project='mlops-assignment2',
    name='distilbert-goodreads-run-1',
    config={'model': model_name, 'epochs': 3, ...}
)
```

`TrainingArguments` reports to `wandb`, and the `Trainer` streams loss, evaluation metrics, and model checkpoints automatically.

### 4.3 Evaluation Metrics
The custom `compute_metrics` function returns **accuracy** and **weighted F1** to the `Trainer`:

```python
def compute_metrics(pred):
    labels = pred.label_ids
    preds = pred.predictions.argmax(-1)
    acc = accuracy_score(labels, preds)
    f1 = precision_recall_fscore_support(labels, preds, average='weighted')[2].mean()
    return {'accuracy': acc, 'f1': f1}
```

### 4.4 Hugging Face Hub Deployment
The fine-tuned model and tokenizer are pushed to the Hub under the repository `https://huggingface.co/balamuralipalani/distilbert-goodreads-genres`:

'''python
from huggingface_hub import login
 
# HF_TOKEN was already loaded from Kaggle Secrets (see Task 4)
login(token=HF_TOKEN)
 
# Push model and tokenizer
model.push_to_hub("balamuralipalani/distilbert-goodreads-genres")
tokenizer.push_to_hub("balamuralipalani/distilbert-goodreads-genres")
 
# Record the link in W&B
wandb.init()
wandb.run.summary["huggingface_model"] = \
    "https://huggingface.co/balamuralipalani/distilbert-goodreads-genres"
wandb.finish() '''


### 4.5 Inference from Hub
A demonstration cell loads the deployed model directly from Hugging Face Hub and performs interactive genre prediction on user-supplied book reviews, confirming the deployment is functional.

---

## 5. Evaluation Results

After 3 epochs of fine-tuning, the model was evaluated on the held-out test set.

| **Metric** | **Score** |
|------------|-----------|
| Accuracy   | 0.570     |
| F1 Score   | 0.586     |
| Eval Loss  | 2.404     |


---


## 6. Challenges Faced & Learnings

1. **Kaggle Internet Toggle:** `ConnectionError` when loading Kaggle Secrets. **Resolution:** Explicitly enable Internet in Notebook Settings and restart the session.
2. **Accelerator Error:** Faced accelerator error while training the model. Its due to the token exhaustion for distilbert which supports maximum of 30552 tokens only **Resolution:** By mistake, i have copied the training model for 2 times in various places. commenting one of the trainer call resolved this problem.
3. **WandB Reporting issue**: First couple of times my executioin results were not reported to WandB dashboard. **Resultion**: There was a command under training arguments was mentioned as report_to=[], updated this to report_to='wandb' and started seeing my dashboard.

---

## 7. Conclusion

This project successfully demonstrates the complete MLOps pipeline on the professor's Goodreads genre-classification task: data preparation, model fine-tuning with live experiment tracking, evaluation with artifact logging, and deployment to a public model repository. All components are reproducible from the tracked W&B configuration and the versioned Hugging Face Hub model.

---

## 8. Submission Links

| Resource | Public Link |
|----------|-------------|
| **GitHub Repository** | https://github.com/balamuralipalani/ML_Ops |
| **Kaggle Notebook** | https://www.kaggle.com/code/balamuralipalaniyand/mlops-assignment-2-fine-tuning-classification |
| **Hugging Face Model** | https://huggingface.co/balamuralipalani/distilbert-goodreads-genres |
| **W&B Dashboard** | https://wandb.ai/g25ait2070-indian-institute-of-technology-jodhpur/mlops-assignment2 |

---
