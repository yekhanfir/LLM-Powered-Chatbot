LLama Chatbot with Fine-tuning
A comprehensive solution for fine-tuning LLama models and deploying them as a chatbot. This project includes data preparation, model fine-tuning, and a web interface for interacting with the fine-tuned model.
🌟 Features

Data preparation pipeline supporting multiple dataset formats
Fine-tuning pipeline for LLama models using PEFT/LoRA
Flask-based web interface for chatbot interaction
Support for custom instructions and model configurations
Wandb integration for training monitoring
CUDA-optimized inference

📋 Prerequisites

Python 3.8+
CUDA-compatible GPU
Anaconda/Miniconda
Hugging Face account with access token
Weights & Biases account (for training monitoring)

🔧 Installation

Clone the repository:

bashCopygit clone [repository-url]
cd llama_chatbot

Create and activate the conda environment:

bashCopyconda env create -f environment.yml
conda activate finetune_llama

Configure the application:

Copy and fill in the configuration files:

finetuning_config.yml (for model fine-tuning)
app/config.yml (for chatbot deployment)
data/data_generation_config.yml (for data preparation)





📊 Data Preparation

Configure your dataset sources in data/data_generation_config.yml:

yamlCopydataset_config:
  hf_dataset_names: # Add your Hugging Face dataset names
  output_path: # Specify output path for processed data

Run the data generation script:

bashCopypython data/generate_data.py
🚀 Fine-tuning

Configure the fine-tuning parameters in finetuning_config.yml:

yamlCopygeneral_config:
  hf_token: # Your Hugging Face token
  wandb_token: # Your Weights & Biases token
training_config:
  data_path: # Path to your processed data
  instruction: # System instruction for the model

Launch the fine-tuning process:

bashCopybash launch_llama_finetuning.sh
Fine-tuning parameters can be adjusted in launch_llama_finetuning.sh. Current configuration:

Base model: meta-llama/Llama-3.2-1B-instruct
Learning rate: 2.0e-4
Training epochs: 10
Batch size: 2
LoRA configuration: r=32, alpha=16

💬 Chatbot Deployment

Configure the chatbot in app/config.yml:

yamlCopygeneral_config:
  hf_access_token: # Your Hugging Face token
model_config:
  model_name: # Base model name
  new_model: # Path to fine-tuned model
chat_config:
  instruction: # System instruction for chat

Run the Flask application:

bashCopycd app
python app.py
The chatbot will be available at http://localhost:5000.
🏗️ Project Structure
Copy.
├── app/
│   ├── app.py              # Flask web application
│   ├── llama_inference.py  # Model inference logic
│   ├── config.yml         # Chatbot configuration
│   └── requirements.txt   # Python dependencies
├── data/
│   ├── generate_data.py   # Data processing script
│   └── data_utils.py      # Data utility functions
├── environment.yml        # Conda environment specification
├── finetuning_config.yml  # Fine-tuning configuration
├── launch_llama_finetuning.sh  # Training launch script
└── llama_finetuning.py   # Fine-tuning implementation
📝 Notes

The application uses PEFT/LoRA for efficient fine-tuning
Training progress can be monitored through Weights & Biases
Maximum sequence length is set to 1024 tokens
Response generation is limited to 150 tokens

🔒 Security
Remember to keep your configuration files and tokens secure. Never commit them to version control.
📄 License
[Add your license information here]
