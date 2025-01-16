# LLM Powered Chatbot Web App Documentation

This repository contains the code for a chatbot powered by the Llama large language model, but you can choose any open-source source LLM, and specialize it to your data, conversational style, etc ... 

# Functionality

The chatbot takes user input and generates a response using a fine-tuned Llama model.

# Here's a breakdown of the repository structure:

* **environment.yml:** Defines the conda environment used for running the project.
* **finetuning_config.yml:** Configuration file for fine-tuning the Llama model.
* **launch_llama_finetuning.sh:** Bash script to launch Llama fine-tuning.
* **llama_finetuning.py:** Python script that performs Llama fine-tuning.
* **app/app.py:** Flask application that serves the chatbot interface.
* **app/config.yml:** Configuration file for the Flask application.
* **app/llama_inference.py:** Script that handles user input and generates chatbot responses using the fine-tuned Llama model.
* **app/requirements.txt:** Lists the Python dependencies required for the Flask application.
* **data/data_generation_config.yml:** Configuration for generating chatbot training data.
* **data/data_utils.py:** Utility functions for data processing.
* **data/generate_data.py:** Script that generates chatbot training data from multiple sources.


# Setting Up the Environment:

1.  **Create a conda environment**:
    ```bash
    conda env create -f environment.yml
    ```
2.  **Activate the environment**:
    ```bash
    conda activate finetune_llama
    ```

# Fine-tuning the Llama Model:

1.  **Update `finetuning_config.yml`**:
    *  Fill in the following fields:
        * `hf_token`: Your Hugging Face Access Token
        * `wandb_token`: Your Weights & Biases API Key (optional)
        * `data_path`: Path to your chatbot training data (JSON format)
        * `instruction`: Starting instruction for the chatbot conversation.
2.  **Run the fine-tuning script**:
    ```bash
    bash launch_llama_finetuning.sh
    ```

# Running the Chatbot Application:

1.  **Update `app/config.yml`**:
    *  Fill in the following fields:
        * `hf_access_token`: Your Hugging Face Access Token
        * `model_config.model_name`: Name of the fine-tuned Llama model (should be the output directory from fine-tuning)
        * `chat_config.instruction`: Starting instruction for the chatbot conversation.
2.  **Run the Flask application**:
    ```bash
    python app/app.py
    ```

This will start the chatbot server, typically accessible at `http://127.0.0.1:5000/` in your web browser.

# Generating Training Data:

* This project provides a script (`data/generate_data.py`) for generating chatbot training data from multiple Hugging Face datasets. 
* Update `data/data_generation_config.yml` with your desired Hugging Face dataset names and output path.
* Update `python data/generate_data.py` to adapt the preprocessing to your chosen datasets.

**Note:** This script is an example and might require modification depending on your specific datasets.
