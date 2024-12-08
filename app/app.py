import sys
sys.path.append('../trl')
from flask import Flask, render_template, request
import requests
from datetime import datetime
import subprocess
# from llama_inference import generate_response

app = Flask(__name__)

FASTAPI_URL = "http://127.0.0.1:8000/api"
conversation = []
prompt_history = '<s> '

@app.route("/", methods=["GET", "POST"])
def chat():
    global conversation
    global prompt_history

    if request.method == "POST":
        raw_prompt = request.form["message"]
        processed_prompt = '[INST] '+ raw_prompt +' [/INST] '
        prompt_history = prompt_history + processed_prompt
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

        # Add user's message
        conversation.append({"sender": "user", "text": raw_prompt, "date": timestamp})

        # Call FastAPI backend
        response = "llm generated response" #generate_response(prompt_history)
        bot_response = response['response']
        prompt_history = prompt_history + bot_response.strip('</s>')

        # Add bot's response
        conversation.append({"sender": "bot", "text": bot_response, "date": timestamp})

    return render_template("chat.html", conversation=conversation)



if __name__ == "__main__":
    app.run()
    fastapi_process = subprocess.Popen(["uvicorn", "app:app", "--port", "8000", "--reload"])
    fastapi_process.wait()
