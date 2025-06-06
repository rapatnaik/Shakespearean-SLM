# -*- coding: utf-8 -*-
"""Copy of rakshan base.ipynb

Automatically generated by Colab.

Original file is located at
    https://colab.research.google.com/drive/1c2zW7Cjj19_z9r9PKWTpxcBpvn5XKi2z
"""

from transformers import GPT2LMHeadModel, GPT2Tokenizer, Trainer, TrainingArguments, DataCollatorForLanguageModeling
from datasets import load_dataset
import torch
import os
import re

tokenizer = GPT2Tokenizer.from_pretrained("Iacopo/Shakespear-GPT2")
tokenizer.pad_token = tokenizer.eos_token  # Fix padding issue

model = GPT2LMHeadModel.from_pretrained("Iacopo/Shakespear-GPT2")
model.resize_token_embeddings(len(tokenizer))

completion_data = [
    "May thy tongue cleave to the roof of thy mouth in lies unspoken.",
    "Let maggots feast upon thy pride ere it sprouts again.",
    "May crows peck wisdom from thy skull, were it ever there.",
    "Thy honor is as hollow as a fool’s purse.",
    "May thy bedsheets be forever damp and thy soup forever cold.",
    "Let the stars forget thy name and the moon shun thy gaze.",
    "May thy shadow betray thee in every dark alley.",
    "Cursed be thy footsteps; may each one lead thee to misfortune.",
    "May goats mock thee with finer speech than thine own.",
    "Let silence be thine only companion in rooms once filled with cheer."
]

with open("completion_data.txt", "w") as f:
    for line in completion_data:
        f.write(line + "\n")
    f.write("\n".join(completion_data * 50))  # 500 lines total

dataset = load_dataset("text", data_files={"train": "completion_data.txt"})

def tokenize_fn(example):
    return tokenizer(example["text"], truncation=True, padding="max_length", max_length=64)

tokenized_dataset = dataset.map(tokenize_fn, batched=True)
train_dataset = tokenized_dataset["train"]

os.environ["WANDB_DISABLED"] = "true"

data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)

training_args = TrainingArguments(
    output_dir="./completion_model",
    num_train_epochs=3,
    per_device_train_batch_size=2,
    logging_steps=10,
    save_steps=0,
    overwrite_output_dir=True,
    save_total_limit=1,
    save_strategy="epoch"
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    data_collator=data_collator
)

trainer.train()
trainer.save_model("./completion_model")
tokenizer.save_pretrained("./completion_model")

def complete_sentence(seed="When the stars vanish from the sky,", model_path="./completion_model"):
    tokenizer = GPT2Tokenizer.from_pretrained("Iacopo/Shakespear-GPT2")
    tokenizer.pad_token = tokenizer.eos_token
    model = GPT2LMHeadModel.from_pretrained(model_path)
    model.eval()

    input_ids = tokenizer.encode(seed, return_tensors="pt")
    with torch.no_grad():
        output = model.generate(
            input_ids,
            max_length=80,
            temperature=0.9,
            top_p=0.95,
            top_k=50,
            do_sample=True,
            no_repeat_ngram_size=4,
            repetition_penalty=1.2,
            pad_token_id=tokenizer.eos_token_id
        )

    output_text = tokenizer.decode(output[0], skip_special_tokens=True)

    # Truncate after the first complete sentence
    match = re.search(r"(.+?\.)", output_text[len(seed):])  # exclude seed from match
    if match:
        return seed.strip() + " " + match.group(1).strip()
    else:
        return output_text.strip()



print(complete_sentence("Where art thou"))

