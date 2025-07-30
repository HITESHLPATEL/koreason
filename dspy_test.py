import json
from bs4 import BeautifulSoup
from tqdm import tqdm
import os
from vllm import LLM, SamplingParams
from datasets import load_dataset
import torch
import pandas as pd
from more_itertools import batched
import glob

# -------- Setup output folder and progress tracker --------
OUTPUT_DIR = "outputs_data"
LOG_FILE = os.path.join(OUTPUT_DIR, "progress.log")
os.makedirs(OUTPUT_DIR, exist_ok=True)

def load_completed_indices():
    if os.path.exists(LOG_FILE):
        with open(LOG_FILE) as f:
            return {int(line.strip()) for line in f if line.strip().isdigit()}
    return set()

def log_progress(idx):
    with open(LOG_FILE, "a") as f:
        f.write(f"{idx}\n")

# -------- Helper to parse outputs safely --------
def safe_parse(output, i):
    try:
        return output.outputs[i].text
    except:
        return None

# -------- Load model --------
model_name = "Qwen/Qwen3-32B"
print("Loading model...")
model = LLM(model=model_name, tensor_parallel_size=torch.cuda.device_count())
tokenizer = model.get_tokenizer()

# -------- Load dataset --------
print("Loading dataset...")
df = load_dataset(
    'amphora/korean-stem',
    token='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',split='train'
                 ).to_pandas()


# -------- Prepare sampling parameters --------
params = SamplingParams(
    temperature=0.6,
    top_p=0.95,
    top_k=20,
    min_p=0,
    max_tokens=16384,
    n=8
)

# -------- Get completed indices to skip --------
completed = load_completed_indices()

# -------- Main Loop --------
print("Starting generation loop...")
idx = 0
for batch in tqdm(batched(df.instruction.values, 500), total=(len(df) + 499) // 500):
    if idx in completed:
        idx += 1
        continue

    qrys = [
        tokenizer.apply_chat_template([
            {"role": "user", "content": 'Think carefully, after you finish thinking, state your answer in fluent and coherent Korean.'},
            {"role": "user", "content": query}
        ], tokenize=False, add_generation_prompt=True, enable_thinking=True)
        for query in batch
    ]

    print(f"Generating batch {idx}...")
    outputs = model.generate(qrys, params)

    dfs = []
    for i in range(params.n):  # params.n = 8
        responses = [safe_parse(out, i) for out in outputs]
        dfs.append(pd.DataFrame({'query': batch, 'responses': responses}))

    save = pd.concat(dfs)
    save_path = os.path.join(OUTPUT_DIR, f'ko-stem-{idx}.csv')
    save.to_csv(save_path, index=False)
    log_progress(idx)
    print(f"Saved: {save_path}")
    idx += 1
