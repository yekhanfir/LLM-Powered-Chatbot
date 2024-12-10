import os
from datasets import load_dataset, concatenate_datasets, Dataset
from transformers import AutoTokenizer
from huggingface_hub import login
from trl import (
    ModelConfig,
    ScriptArguments,
    SFTConfig,
    SFTTrainer,
    TrlParser,
    get_kbit_device_map,
    get_peft_config,
    get_quantization_config,
)
from data.data_utils import load_dataset, filter_by_length
import yaml
HF_TOKEN=os.environ["HF_TOKEN"]

with open('finetuning_config.yml', 'r') as config_file:
    config = yaml.load(config_file)

def format_row(row):
    """_summary_

    Args:
        tokenizer (_type_): _description_
        row (_type_): _description_
        instruction (_type_): _description_

    Returns:
        _type_: _description_
    """
    instruction = config['training_config']['instruction']
    row_json = [{"role": "system", "content": instruction },
               {"role": "user", "content": row["text"]["user"]},
               {"role": "assistant", "content": row["text"]["assistant"]}]
    
    row["text"] = tokenizer.apply_chat_template(row_json, tokenize=False)
    return row


if __name__ == "__main__":
    parser = TrlParser((ScriptArguments, SFTConfig, ModelConfig))
    script_args, training_args, model_config = parser.parse_args_and_config()
    login(HF_TOKEN)

    # Initialize model
    quantization_config = get_quantization_config(model_config)
    model_kwargs = dict(
        revision=model_config.model_revision,
        trust_remote_code=model_config.trust_remote_code,
        attn_implementation=model_config.attn_implementation,
        torch_dtype=model_config.torch_dtype,
        use_cache=False if training_args.gradient_checkpointing else True,
        device_map=get_kbit_device_map() if quantization_config is not None else None,
        quantization_config=quantization_config,
    )
    training_args.model_init_kwargs = model_kwargs

    # Initialize tokenizer
    tokenizer = AutoTokenizer.from_pretrained(
        model_config.model_name_or_path, trust_remote_code=model_config.trust_remote_code, use_fast=True
    )
    tokenizer.pad_token = tokenizer.eos_token

    # Initialize dataset, format and filter dataset
    dataset_dict = load_dataset(config['finetuning_config']['data_path'])
    dataset = Dataset.from_dict(dataset_dict)
    dataset = dataset.map(format_row)
    dataset = dataset.filter(filter_by_length)

    # train the model
    trainer = SFTTrainer(
        model=model_config.model_name_or_path,
        args=training_args,
        train_dataset=dataset,
        eval_dataset=dataset if training_args.eval_strategy != "no" else None,
        processing_class=tokenizer,
        peft_config=get_peft_config(model_config),
    )

    trainer.train()

    # Save and push to hub
    trainer.save_model(training_args.output_dir)
    if training_args.push_to_hub:
        trainer.push_to_hub(dataset_name=script_args.dataset_name)
