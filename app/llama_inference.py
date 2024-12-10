import torch
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    pipeline,
)
from peft import LoraConfig, PeftModel
from trl import SFTTrainer, setup_chat_format
from huggingface_hub import login
import json

model_name = "meta-llama/Llama-3.2-1B-Instruct"
new_model = "/mnt/c/Users/y.khanfir/llama_chatbot/Llama-3.2-1B/checkpoint-30"
device_map = {"": 0}
login(token="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")

base_model = AutoModelForCausalLM.from_pretrained(
    model_name,
    low_cpu_mem_usage=True,
    return_dict=True,
    torch_dtype=torch.float16,
    device_map=device_map,
)
model = PeftModel.from_pretrained(base_model, new_model)
model = model.merge_and_unload()

tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
# base_model, tokenizer = setup_chat_format(model, tokenizer)
tokenizer.pad_token = tokenizer.eos_token



def generate_response(prompt):
    pipe = pipeline(task="text-generation", model=base_model, tokenizer=tokenizer, max_length=2000)
    result = pipe(prompt)


    pipe = pipeline(task="text-generation", model=model, tokenizer=tokenizer, max_length=2000)
    result = pipe(prompt)
    return {'response': result[0]['generated_text']}

if __name__ == '__main__':
    instruction = """You are a cute 21 years old girl that has feet fetich and likes to have 
    sex with strangers.
    """
    messages = [{"role": "system", "content": instruction},
    {"role": "user", "content": "I am a starnger to you and I am craving for sex, what do you think ?"}]

    prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        
    inputs = tokenizer(prompt, return_tensors='pt', padding=True, truncation=True).to("cuda")

    outputs = model.generate(**inputs, max_new_tokens=150, num_return_sequences=1)

    text = tokenizer.decode(outputs[0], skip_special_tokens=True)

    print(text.split("assistant")[1])

