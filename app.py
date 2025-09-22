import openai
from flask import Flask, request, jsonify
from flask_cors import CORS
from dotenv import load_dotenv
import os
import logging
from logging.handlers import RotatingFileHandler
from datetime import datetime
import time
import json
import re
import tiktoken
from tenacity import retry, wait_random_exponential, stop_after_attempt, retry_if_exception_type
from openai.error import RateLimitError, APIError, Timeout, ServiceUnavailableError
from werkzeug.exceptions import HTTPException
from flask_limiter import Limiter
from flask_limiter.util import get_remote_address

# Load environment variables from .env file
load_dotenv()
app = Flask(__name__)
CORS(app)
limiter = Limiter(get_remote_address, app=app, default_limits=["200 per day", "50 per hour"])
openai.api_key = os.getenv("OPENAI_API_KEY")
MODEL = os.getenv("MODEL", "gpt-3.5-turbo")
MAX_TOKENS = int(os.getenv("MAX_TOKENS", 4096))
TEMPERATURE = float(os.getenv("TEMPERATURE", 0.7))
LOG_FILE = os.getenv("LOG_FILE", "app.log")
MAX_RETRIES = int(os.getenv("MAX_RETRIES", 5))
SYSTEM_PROMPT = os.getenv("SYSTEM_PROMPT", "You are a helpful assistant.")

# Set up logging
handler = RotatingFileHandler(LOG_FILE, maxBytes=1000000, backupCount=3)
handler.setLevel(logging.INFO)
formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
handler.setFormatter(formatter)
app.logger.addHandler(handler)
app.logger.setLevel(logging.INFO)

# Initialize tokenizer
tokenizer = tiktoken.encoding_for_model(MODEL)

def num_tokens_from_messages(messages, model=MODEL):
    """Returns the number of tokens used by a list of messages."""
    try:
        encoding = tiktoken.encoding_for_model(model)
    except KeyError:
        encoding = tiktoken.get_encoding("cl100k_base")
    if model.startswith("gpt-3.5-turbo"):
        tokens_per_message = 4
        tokens_per_name = -1
    elif model.startswith("gpt-4"):
        tokens_per_message = 3
        tokens_per_name = 1
    else:
        raise NotImplementedError(f"num_tokens_from_messages() is not implemented for model {model}.")
    
    num_tokens = 0
    for message in messages:
        num_tokens += tokens_per_message
        for key, value in message.items():
            num_tokens += len(encoding.encode(value))
            if key == "name":
                num_tokens += tokens_per_name
    num_tokens += 3
    return num_tokens

@retry(wait=wait_random_exponential(min=1, max=60), stop=stop_after_attempt(MAX_RETRIES), retry=retry_if_exception_type((RateLimitError, APIError, Timeout, ServiceUnavailableError)))
def call_openai_api(messages):
    response = openai.ChatCompletion.create(
        model=MODEL,
        messages=messages,
        temperature=TEMPERATURE,
        max_tokens=MAX_TOKENS - num_tokens_from_messages(messages),
        n=1,
        stop=None,
    )
    return response
@app.errorhandler(HTTPException)
def handle_http_exception(e):
    response = e.get_response()
    response.data = json.dumps({
        "code": e.code,
        "name": e.name,
        "description": e.description,
    })
    response.content_type = "application/json"
    return response
@app.errorhandler(Exception)
def handle_exception(e):
    app.logger.error(f"Unhandled exception: {str(e)}")
    return jsonify(error="Internal Server Error"), 500
@app.route('/chat', methods=['POST'])
@limiter.limit("10 per minute")

def chat():
    start_time = time.time()
    client_ip = request.remote_addr
    try:
        data = request.get_json()
        if not data or 'messages' not in data:
            return jsonify(error="Invalid request, 'messages' field is required."), 400
        
        user_messages = data['messages']
        if not isinstance(user_messages, list) or not all(isinstance(msg, dict) and 'role' in msg and 'content' in msg for msg in user_messages):
            return jsonify(error="'messages' must be a list of dictionaries with 'role' and 'content'."), 400
        
        messages = [{"role": "system", "content": SYSTEM_PROMPT}] + user_messages
        total_tokens = num_tokens_from_messages(messages)
        
        if total_tokens > MAX_TOKENS:
            return jsonify(error=f"Message exceeds maximum token limit of {MAX_TOKENS}. Current tokens: {total_tokens}"), 400
        
        response = call_openai_api(messages)
        assistant_message = response.choices[0].message['content']
        response_tokens = response.usage.total_tokens
        end_time = time.time()
        
        log_entry = {
            "timestamp": datetime.utcnow().isoformat(),
            "client_ip": client_ip,
            "request": messages,
            "response": assistant_message,
            "total_tokens": total_tokens,
            "response_tokens": response_tokens,
            "duration_seconds": end_time - start_time
        }
        app.logger.info(json.dumps(log_entry))
        
        return jsonify(assistant_message=assistant_message, total_tokens=total_tokens, response_tokens=response_tokens)
    
    except Exception as e:
        app.logger.error(f"Error processing request from {client_ip}: {str(e)}")
        return jsonify(error="Internal Server Error"), 500
if __name__ == '__main__':
    app.run(host='xx', port=5000), debug=False)
    app.run(host='y', port=5000), debug=False)


    