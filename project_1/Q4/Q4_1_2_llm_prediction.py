import pandas as pd
from ollama import chat
from sklearn.metrics import roc_auc_score, average_precision_score
import numpy as np
import re
from concurrent.futures import ThreadPoolExecutor, as_completed
from itertools import product

# Data Import
summaries_a = pd.read_csv("summaries_a.csv")
summaries_b = pd.read_csv("summaries_b.csv")
summaries_c = pd.read_csv("summaries_c.csv")

# Setting Running Parameters
run_efficient = False
#llm_model = 'deepseek-r1:7b'
#llm_model = 'llama3.2:3b'
llm_model = 'gemma2:9b'
device_map = "cpu" # or "cuda"

#----------------------- SYSTEM PROMPTS ---------------------

system_prompt_binary = """You are a clinical risk prediction assistant designed to assess the likelihood of in-hospital mortality based on structured patient summaries derived from ICU time-series data. 

You will be provided with a concise, pre-processed text description of a patient's vital signs and lab measurements recorded during the first 48 hours of their ICU stay. Your task is to classify whether the patient is at risk of dying during their hospital stay.

Respond ONLY with a single number:
- `1` if the patient is at high risk of death.
- `0` if the patient is likely to survive.

Do not provide any explanations, reasoning, or text. Only return the number `0` or `1`.
"""

system_prompt_score = """You are a clinical risk prediction assistant designed to assess the likelihood of in-hospital mortality based on structured patient summaries derived from ICU time-series data. 

You will be provided with a concise, pre-processed text description of a patient's vital signs and lab measurements recorded during the first 48 hours of their ICU stay. Your task is to determine the patient risk of dying during their hospital stay.

Your goal is to assess the patient's risk of in-hospital mortality on a scale from 1 to 10:
- `1` means extremely low risk (highly likely to survive).
- `10` means extremely high risk (very likely to die in hospital).

Respond with a single integer between 1 and 10. Do not include explanations, context, or additional output. Return only the number.
"""

#-------------------- OLLAMA PROCESSING -------------------------

def clean_llm_output(text):
    """
    Cleans LLM output by removing <think> blocks and 'Answer:' prefix. (relevant for DeepSeek)
    """
    # Remove <think>...</think> block
    text = re.sub(r'<think>.*?</think>', '', text, flags=re.DOTALL | re.IGNORECASE)

    # Remove 'Answer:' prefix (case-insensitive, any surrounding whitespace)
    text = re.sub(r'(?i)\banswer\s*:\s*', '', text)

    # Strip leading/trailing whitespace
    return text.strip()


# ---- Utility: Extract score or binary ---- #
def extract_score_from_response(text, mode="score"):
    """
    Extracts a number from model output or interprets binary label.
    """
    text = text.lower()
    text = clean_llm_output(text)

    if mode == "score":
        match = re.search(r'(\d+(\.\d+)?)', text)
        if match:
            val = float(match.group(1))
            return min(max(val, 1.0), 10.0) / 10.0  # normalize 1-10 to 0-1

    else:  # binary mode
        # Try extracting a number first
        match = re.search(r'(\d+(\.\d+)?)', text)
        if match:
            val = float(match.group(1))
            # Normalize 0–100 or 1–10 range to 0–1 if needed
            if val == 1.0 or val == 0.0:
                return val
            else:
                print(f"Binary response didn't work: {text}")

        # Fallback to keyword heuristics
        if any(word in text for word in ["yes", "high", "die", "likely"]):
            return 1.0
        elif any(word in text for word in ["no", "low", "survive", "unlikely"]):
            return 0.0

    return 0.5  # fallback neutral

# ---- Core inference function ---- #

def query_llm(summary_text, mode="score", few_shot_context=None):
    """
    Calls llm_model model via ollama.chat using user prompt.
    """
    user_prompt = f"{few_shot_context}\n\n### Task \nInput:\n{summary_text}\nAnswer:" if few_shot_context else f"Input:\n{summary_text}\nAnswer:"

    response = chat(model=llm_model, messages=[{
        'role': 'system',
        'content': system_prompt_score if mode == "score" else system_prompt_binary,
    },
        {
            'role': 'user',
            'content': user_prompt
        }])

    return response['message']['content']


# ---------------------------------- PROMPT CREATION AND RESULT EVALUATION ---------------


# ---- Few-shot formatter ---- #

def generate_few_shot_examples(df_train, text_col="summary_statistical", label_col="In-hospital_death", max_examples=3,
                               mode="score"):
    """
    Creates few-shot context examples from labeled training set.
    """
    examples = df_train.sample(n=max_examples, random_state=42)
    formatted = "### Examples\n"
    for _, row in examples.iterrows():
        label = row[label_col]
        label_text = f"{int(label * 9 + 1)}/10" if mode == "score" else f"{label}"
        formatted += f"Input:\n{row[text_col]}\nAnswer: {label_text}\n\n"
    return formatted.strip()

# ---- Full evaluation runner ---- #

def evaluate_llm_predictions(summary_df, text_col='summary_statistical', label_col="In-hospital_death",
                             mode="score", few_shot=False, df_train_for_examples=None, max_workers=4):
    """
    Runs inference across all rows in summary_df using parallel threads.
    """
    labels = summary_df[label_col].values
    summaries = summary_df[text_col].values

    # Build few-shot context once
    context = None
    if few_shot and df_train_for_examples is not None:
        context = generate_few_shot_examples(df_train_for_examples, text_col, mode=mode)

    def process_summary(summary):
        llm_output = query_llm(summary, mode, few_shot_context=context)
        return extract_score_from_response(llm_output, mode=mode)

    # Run in parallel
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = [executor.submit(process_summary, s) for s in summaries]
        predictions = [f.result() for f in as_completed(futures)]

    # Sort predictions to maintain order (as_completed gives unordered results)
    predictions_sorted = [None] * len(futures)
    for i, f in enumerate(futures):
        predictions_sorted[i] = f.result()

    predictions = np.array(predictions_sorted)

    try:
        auroc = roc_auc_score(labels, predictions)
        auprc = average_precision_score(labels, predictions)
    except Exception as e:
        print("[!] Metric calculation error:", e)
        auroc = None
        auprc = None

    return {
        "predictions": predictions,
        "true_labels": labels,
        "auroc": auroc,
        "auprc": auprc
    }

# --------------------------- EXECUTION -----------------

summaries_c_small = summaries_c.sample(100, random_state = 1)
summary_cols = ['summary_statistical', 'summary_trend']
prompt_modes = ['few-shot', 'zero-shot']
score_modes = ['binary', 'score']

# Loop through all combinations
for summary_col, prompt_mode, score_mode in product(summary_cols, prompt_modes, score_modes):
    kwargs = {
        'summary_df': summaries_c_small if run_efficient else summaries_c,
        'text_col': summary_col,
        'mode': score_mode,
    }

    label = f"{prompt_mode.title()} | {summary_col} | {score_mode}"

    # Add few-shot parameters if needed
    if prompt_mode == 'few-shot':
        kwargs['few_shot'] = True
        kwargs['df_train_for_examples'] = summaries_a

    results = evaluate_llm_predictions(**kwargs)
    print(f"{label} → AUROC: {results['auroc']:.3f}, AUPRC: {results['auprc']:.3f}")