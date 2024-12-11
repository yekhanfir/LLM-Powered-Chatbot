from flask import Flask, render_template, request
import requests

app = Flask(__name__)

@app.route("/", methods=["GET", "POST"])
def chat():
    conversation = ""
    inference_server_URL = " https://09f9-160-158-112-160.ngrok-free.app/inference"
    if request.method == "POST":
        raw_prompt = request.form["message"]
        response = requests.post(
            inference_server_URL+"?message="+raw_prompt, 
        ).json()
        conversation = response['response']
    return render_template("chat.html", conversation=conversation)

if __name__ == "__main__":
    app.run() # app.run(port=8000)