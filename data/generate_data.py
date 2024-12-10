import requests
import json
import pandas as pd
from datasets import load_dataset
import http.server
import socketserver
import os

conversation_id = 1  # Start conversation numbering from (index_discussion = 1)
transformed_dataset1 = {} # dataset 1 = Maxx0/sexting-nsfw-adultconten = 2156 lines (1 discussion)
transformed_dataset2 = {} # dataset 2 = freQuensy23/sexting_prompts = 1076 (1 discussion)
transformed_dataset3 = {} # dataset 3 = Chadgpt-fam/sexting_dataset =  1436 lines (10 discussions)
transformed_dataset4 = {} # dataset 4 = SPACENOS/exbot-nsfw-sexting = 10212 (1 discussion)

def transform_dataset1(dataset):
    global conversation_id
    conversation = []
    for index in range(len(dataset)):
        he_content = dataset['He'].iloc[index]
        she_content = dataset['She'].iloc[index]
        if pd.notna(he_content):
            conversation.append({"role": "user", "content": he_content})
        if pd.notna(she_content):
            conversation.append({"role": "assistant", "content": she_content})
        transformed_dataset1[str(conversation_id)] = conversation
    conversation_id += 1
    return transformed_dataset1

dataset1 = load_dataset("Maxx0/sexting-nsfw-adultconten", split="train")
df1 = pd.DataFrame(dataset1)
transformed_dataset1 = transform_dataset1(df1)


def transform_dataset2(dataset):
    global conversation_id
    conversation = []
    for index in range(len(dataset)):
        he_content = dataset['answer'].iloc[index]
        she_content = dataset['prompt'].iloc[index]
        last_user_segment = she_content.split("USER:")[-1]
        user_part = last_user_segment.split("\nASSISTANT:")[0].strip().replace("\n", ", ")
        final_she_prompt = user_part
        final_he_answer = he_content.replace("</s>", "").strip()
        if pd.notna(final_she_prompt):
            conversation.append({"role": "assistant", "content": final_she_prompt})
        if pd.notna(final_he_answer):
            conversation.append({"role": "user", "content": final_he_answer})
        transformed_dataset2[str(conversation_id)] = conversation
    conversation_id += 1
    return transformed_dataset2

dataset2 = load_dataset("freQuensy23/sexting_prompts", split="train")
df2 = pd.DataFrame(dataset2)
transformed_dataset2 = transform_dataset2(df2)


def transform_dataset3(dataset):
    global conversation_id
    for index in range(len(dataset)):
        conversation = []
        text_content = dataset['text'].iloc[index]
        segments = text_content.split("[INST]")
        for i in range(1, len(segments)):
            segment = segments[i]
            if "[/INST]" in segment:
                he_content = segment.split("[/INST]")[0].strip()
                she_content_part = segment.split("[/INST]")[1] 
                if "</s>" in she_content_part:
                    she_content = she_content_part.split("</s>")[0].strip()
                elif "[INST]" in she_content_part:
                    she_content = she_content_part.split("[INST]")[0].strip()
                else:
                    she_content = she_content_part.strip()
                if he_content:
                    conversation.append({"role": "user", "content": he_content})
                if she_content:
                    conversation.append({"role": "assistant", "content": she_content})
        transformed_dataset3[str(conversation_id)] = conversation
        conversation_id += 1
    return transformed_dataset3

dataset3 = load_dataset("Chadgpt-fam/sexting_dataset", split="train")
df3 = pd.DataFrame(dataset3)
transformed_dataset3 = transform_dataset3(df3)


def transform_dataset4(dataset):
    global conversation_id
    conversation = []
    for index in range(len(dataset)):
        he_content = dataset['Boy'].iloc[index]
        she_content = dataset['Girl'].iloc[index]
        if pd.notna(she_content):
            conversation.append({"role": "assistant", "content": she_content})
        if pd.notna(he_content):
            conversation.append({"role": "user", "content": he_content})
        transformed_dataset4[str(conversation_id)] = conversation
    conversation_id += 1
    return transformed_dataset4

dataset4 = load_dataset("SPACENOS/exbot-nsfw-sexting", split="train")
df4 = pd.DataFrame(dataset4)
transformed_dataset4 = transform_dataset4(df4)

final_combined_dataset = {**transformed_dataset1, **transformed_dataset2, **transformed_dataset3, **transformed_dataset4}


def transform_dataset(data):
    transformed_data = {
        "id": [],
        "text": []
    }

    current_id = 1

    for conversation in data.values():
        for i in range(0, len(conversation) - 1, 2):  # Process pairs of messages
            user_message = conversation[i]
            assistant_message = conversation[i + 1] if i + 1 < len(conversation) else None

            if assistant_message:  # Ensure both user and assistant messages exist
                transformed_data["id"].append(current_id)
                transformed_data["text"].append({
                    "user": user_message["content"],
                    "assistant": assistant_message["content"]
                })
                current_id += 1

    return transformed_data


final_json = json.dumps(transform_dataset(final_combined_dataset), indent=4)
json_filename = "dataset.json"

# Write the final result to a file
with open(json_filename, "w") as f:
    f.write(final_json)

 