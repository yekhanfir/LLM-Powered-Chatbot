import os
import torch
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
)
from peft import PeftModel
from huggingface_hub import login
import yaml

with open('app/config.yml', 'r') as config_file:
    config = yaml.load(config_file)

model_name = config['model_config']['model_name']
new_model = config['model_config']['new_model']

device_map = {"": 0}
login(token=config['general_config']['hf_access_token'])

# load base model
base_model = AutoModelForCausalLM.from_pretrained(
    model_name,
    low_cpu_mem_usage=True,
    return_dict=True,
    torch_dtype=torch.float16,
    device_map=device_map,
)
# load finetuned model from checkpoint
model = PeftModel.from_pretrained(base_model, new_model)
# merge peft weights from checkpoint into base model
model = model.merge_and_unload()

# initialize tokenizer
tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
tokenizer.pad_token = tokenizer.eos_token


def generate_response(prompt):
    """_summary_

    Args:
        prompt (_type_): _description_

    Returns:
        _type_: _description_
    """
    prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = tokenizer(prompt, return_tensors='pt', padding=True, truncation=True).to("cuda")
    outputs = model.generate(**inputs, max_new_tokens=150, num_return_sequences=1)
    text = tokenizer.decode(outputs[0], skip_special_tokens=True)
    return {'response': text.split("assistant")[1]}


if __name__ == '__main__':
    instruction = config['chat_config']['instruction']
    messages = [{"role": "system", "content": instruction},
    {"role": "user", "content": "I am a starnger to you and I am craving for sex, what do you think ?"}]

    prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        
    inputs = tokenizer(prompt, return_tensors='pt', padding=True, truncation=True).to("cuda")

    outputs = model.generate(**inputs, max_new_tokens=150, num_return_sequences=1)

    text = tokenizer.decode(outputs[0], skip_special_tokens=True)

    print(text.split("assistant")[1])

