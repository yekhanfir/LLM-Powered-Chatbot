from flask import Flask, render_template, request
from datetime import datetime
from llama_inference import generate_response
import yaml
import requests

with open('app/config.yml', 'r') as config_file:
    config = yaml.load(config_file)

app = Flask(__name__)

conversation = []
instruction = config['chat_config']['instruction']

prompt_history = [{"role": "system", "content": instruction }]

@app.route("/", methods=["GET", "POST"])
def end_to_end_chat():
    global conversation
    global prompt_history

    if request.method == "POST":
        raw_prompt = request.form["message"]

        prompt_history.append({"role": "user", "content": raw_prompt})
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

        # Add user's message (for display)
        conversation.append({"sender": "user", "text": raw_prompt, "date": timestamp})

        # Call FastAPI backend
        response = generate_response(prompt_history)
        bot_response = response['response']
        prompt_history.append({"role": "assistant", "content": bot_response})

        # Add bot's response
        conversation.append({"sender": "bot", "text": bot_response, "date": timestamp})

    return render_template("chat.html", conversation=conversation)


@app.route("/inference", methods=["GET", "POST"])
def inference():
    global conversation
    global prompt_history

    if request.method == "POST":
        raw_prompt = request.args.get("message")

        prompt_history.append({"role": "user", "content": raw_prompt})
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

        # Add user's message (for display)
        conversation.append({"sender": "user", "text": raw_prompt, "date": timestamp})

        # Call FastAPI backend
        response = generate_response(prompt_history)
        bot_response = response['response']
        prompt_history.append({"role": "assistant", "content": bot_response})
        
        # Add bot's response
        conversation.append({"sender": "bot", "text": bot_response, "date": timestamp})

    return {'response': conversation}

if __name__ == "__main__":
    app.run()