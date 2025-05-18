import streamlit as st
from transformers import GPT2Tokenizer, GPT2LMHeadModel
import torch
import re
import os 

@st.cache_resource
def load_model():
    model_path = os.path.abspath("./completion_model")
    tokenizer = GPT2Tokenizer.from_pretrained(model_path)
    model = GPT2LMHeadModel.from_pretrained(model_path)
    model.eval()
    return tokenizer, model

def complete_sentence(seed, tokenizer, model):
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

    generated_text = tokenizer.decode(output[0], skip_special_tokens=True)
    continuation = generated_text[len(seed):].strip()
    match = re.search(r"(.+?\.)", continuation)
    if match:
        return seed.strip() + " " + match.group(1).strip()
    return generated_text.strip()

# --- Streamlit UI ---
st.title("📝 Shakespearean Sentence Finisher")
st.write("Enter a poetic or dramatic prompt, and let the fine-tuned GPT-2 complete it in Shakespearean style.")

seed = st.text_input("Seed Text", value="Where art thou")

if st.button("Complete Sentence"):
    tokenizer, model = load_model()
    result = complete_sentence(seed, tokenizer, model)
    st.markdown("### 🎭 Completed Sentence")
    st.success(result)
