from datasets import load_dataset

# Replace with your actual Hugging Face token
HF_TOKEN = "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"

# Load the dataset using your token
df = load_dataset(
    'KoReason/ot3-ko-sampled',
    split='train',
    use_auth_token=HF_TOKEN
).to_pandas()

contents = []
for _,row in df.iterrows():
    content = [
        {'content':row.question,'role':'user'},
        {'content':row.response,'role':'assistant'},
    ]
    contents.append(content)
df['messages'] = contents

# df is your pandas DataFrame
df.to_json(
    "train_ot3_ko_sampled.jsonl",   # file name
    orient="records", # one JSON object per line
    lines=True,       # write in JSON Lines format
)
