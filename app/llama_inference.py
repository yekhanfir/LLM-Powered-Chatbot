import torch
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    pipeline,
)
from peft import LoraConfig, PeftModel
from trl import SFTTrainer

model_name = "meta-llama/Llama-3.2-1B"
new_model = "/mnt/c/Users/y.khanfir/Finetuning_lllama/Llama-3.2-1B/checkpoint-3"
device_map = {"": 0}

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
tokenizer.pad_token = tokenizer.eos_token
tokenizer.padding_side = "right"



def generate_response(prompt):
    pipe = pipeline(task="text-generation", model=base_model, tokenizer=tokenizer, max_length=200)
    result = pipe(prompt)


    pipe = pipeline(task="text-generation", model=model, tokenizer=tokenizer, max_length=200)
    result = pipe(prompt)
    return {'response': result[0]['generated_text']}