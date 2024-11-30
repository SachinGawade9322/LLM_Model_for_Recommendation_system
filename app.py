import os
import pandas as pd
from flask import Flask, request, jsonify, render_template
from flask_cors import CORS
from transformers import AutoModelForCausalLM, AutoTokenizer
from llm_model import LLMModel
import torch

app = Flask(__name__)
CORS(app)

model_name = "microsoft/DialoGPT-medium"
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(model_name).to(device)

llm_model = LLMModel() 
df = pd.read_csv('projects_dataset.csv')
df = llm_model.preprocess_dataset(df)
df = llm_model.generate_features(df)

def generate_response(user_input, chat_history_ids):
    new_user_input_ids = tokenizer.encode(user_input + tokenizer.eos_token, return_tensors='pt').to(device)
    bot_input_ids = torch.cat([chat_history_ids, new_user_input_ids], dim=-1) if chat_history_ids is not None else new_user_input_ids
    chat_history_ids = model.generate(bot_input_ids, max_length=1000, pad_token_id=tokenizer.eos_token_id)
    response = tokenizer.decode(chat_history_ids[:, bot_input_ids.shape[-1]:][0], skip_special_tokens=True)
    return response, chat_history_ids

@app.route('/', methods=['GET', 'POST'])
def home():
    if request.method == 'POST':
        return jsonify({"message": "This is a POST request to the root!"}), 200
    return render_template('index.html')

@app.route('/chat', methods=['POST'])
def chat():
    data = request.get_json()
    user_input = data.get('user_input', '').strip()
    chat_history_ids = data.get('chat_history_ids', None)

    if not user_input:
        return jsonify({"error": "No user input provided"}), 400

    if chat_history_ids is not None:
        chat_history_ids = torch.tensor(chat_history_ids).to(device)

    response, chat_history_ids = generate_response(user_input, chat_history_ids)
    return jsonify({"response": response, "chat_history_ids": chat_history_ids.tolist()}), 200

@app.route('/recommend_projects', methods=['POST'])
def recommend_projects():
    data = request.get_json()
    user_query = data.get('user_query', '').strip()
    if not user_query:
        return jsonify({"error": "No user query provided"}), 400

    recommendations = llm_model.recommend_projects(user_query, df, top_n=5)
    recommended_projects = recommendations[['job_description', 'similarity']].to_dict(orient='records')
    return jsonify({"recommendations": recommended_projects}), 200

if __name__ == "__main__":
    app.run(debug=True, port=5000)