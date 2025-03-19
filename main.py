# Standard library imports
import os
import json
import logging

# Third-party imports
from flask import Flask, request, jsonify, Response
from dotenv import load_dotenv
import requests
from flask_session import Session
import google.generativeai as genai

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Load environment variables
load_dotenv()

# Initialize Flask app
app = Flask(__name__)
app.config["SESSION_PERMANENT"] = False
app.config["SESSION_TYPE"] = "filesystem"
Session(app)

# Global configuration - choose AI provider
AI_PROVIDER = os.getenv('AI_PROVIDER', 'openai').lower().strip()  # Options: 'openai' or 'gemini'
logger.info(f"AI_PROVIDER set to: {AI_PROVIDER}")

# Validate required environment variables
required_vars = ['OPENAI_API_KEY'] if AI_PROVIDER == 'openai' else ['GEMINI_API_KEY']
for var in required_vars:
    if not os.getenv(var):
        logger.error(f"Missing required environment variable: {var}")

# Initialize Gemini if selected
if AI_PROVIDER == 'gemini':
    try:
        gemini_api_key = os.getenv('GEMINI_API_KEY')
        if gemini_api_key:
            masked_key = f"{gemini_api_key[:4]}...{gemini_api_key[-4:]}" if len(gemini_api_key) > 8 else "***"
            logger.info(f"Initializing Gemini with API key: {masked_key}")
            genai.configure(api_key=gemini_api_key)
            logger.info("Gemini API configured successfully")
        else:
            logger.error("No Gemini API key found!")
    except Exception as e:
        logger.error(f"Failed to initialize Gemini: {str(e)}")

# Initialize session storage for users
if 'users_state' not in app.config:
    app.config['users_state'] = {}

# Function to clean Alisa greeting
def clean_request(request_text):
    if not request_text:
        return ""
    cut_words = ['Алиса', 'алиса']
    for word in cut_words:
        if request_text.lower().startswith(word.lower()):
            request_text = request_text[len(word):]
    return request_text.strip()

# Function to interact with OpenAI
def ask_openai(message, messages, max_retries=2):
    api_key = os.getenv('OPENAI_API_KEY')
    if not api_key:
        logger.error("OpenAI API key not found")
        return 'Не удалось получить ответ от сервиса (ошибка конфигурации).'
    
    # Create the proper message format
    formatted_messages = []
    for msg in messages[-10:]:  # Limit context window size
        formatted_messages.append({"role": "user", "content": msg})
    
    # Add the current message
    formatted_messages.append({"role": "user", "content": message})
    
    retries = 0
    while retries <= max_retries:
        try:
            response = requests.post(
                'https://api.openai.com/v1/chat/completions',
                headers={
                    'Authorization': f'Bearer {api_key}',
                    'Content-Type': 'application/json'
                },
                json={
                    'model': 'gpt-4o-mini',
                    'messages': formatted_messages,
                    'max_tokens': 1000,
                    'temperature': 0.7
                },
                timeout=30.0
            )
            
            # Check if the response was successful
            response.raise_for_status()
            
            response_data = response.json()
            logger.info("OpenAI API response received")
            
            if 'choices' not in response_data or len(response_data['choices']) == 0:
                logger.error(f"Unexpected API response format: {response_data}")
                return 'Не удалось получить ответ от сервиса.'
                
            return response_data['choices'][0]['message']['content'].strip()
        except requests.exceptions.RequestException as e:
            retries += 1
            logger.error(f"OpenAI API request error (attempt {retries}/{max_retries}): {str(e)}")
            if hasattr(e, 'response') and e.response is not None:
                logger.error(f"Response status: {e.response.status_code}")
                logger.error(f"Response body: {e.response.text}")
            
            if retries > max_retries:
                return 'Не удалось получить ответ от сервиса. Пожалуйста, попробуйте позже.'
        except Exception as e:
            logger.error(f"OpenAI API error: {str(e)}")
            return 'Не удалось получить ответ от сервиса.'

# Function to interact with Gemini
def ask_gemini(message, messages, max_retries=2):
    gemini_api_key = os.getenv('GEMINI_API_KEY')
    if not gemini_api_key:
        logger.error("Gemini API key not found")
        return 'Не удалось получить ответ от сервиса (ошибка конфигурации).'
    
    retries = 0
    while retries <= max_retries:
        try:
            # Combine previous messages and current message with clear distinction
            history_text = ""
            if messages:
                history_text = "Previous conversation:\n" + "\n".join([f"User: {msg}" for msg in messages[-5:]])  # Limit context
            full_prompt = f"{history_text}\n\nCurrent message: {message}" if history_text else message
            
            logger.info(f"Sending request to Gemini API with prompt: {full_prompt[:100]}...")
            
            # Create the model and generate content
            model = genai.GenerativeModel('gemini-1.5-flash')
            response = model.generate_content(full_prompt)
            
            logger.info("Gemini API response received")
            
            # Extract the text from the response
            if response and hasattr(response, 'text'):
                return response.text.strip()
            else:
                logger.error(f"Unexpected Gemini response format: {response}")
                return 'Не удалось получить ответ от сервиса.'
        except Exception as e:
            retries += 1
            logger.error(f"Gemini API error (attempt {retries}/{max_retries}): {str(e)}")
            if retries > max_retries:
                return 'Не удалось получить ответ от сервиса. Пожалуйста, попробуйте позже.'

# Health check endpoint
@app.route('/health', methods=['GET'])
def health_check():
    status = {
        'status': 'ok',
        'ai_provider': AI_PROVIDER
    }
    return jsonify(status)
      
# Main API endpoint
@app.route('/', methods=['POST'])
def handle_request():
    try:
        # Validate request content type
        if not request.is_json:
            logger.warning("Request content type is not application/json")
            return jsonify({
                "response": {
                    "text": "Неверный формат запроса. Ожидается JSON.",
                    "end_session": False
                },
                "version": "1.0"
            }), 400
        
        input_data = request.get_json(force=True)
        
        # Validate required fields
        if not input_data.get('session') or not input_data.get('version'):
            logger.warning("Missing required fields in request")
            return jsonify({
                "response": {
                    "text": "Неверный формат запроса. Отсутствуют обязательные поля.",
                    "end_session": False
                },
                "version": "1.0"
            }), 400
        
        response = {
            'session': input_data['session'],
            'version': input_data['version'],
            'response': {
                'end_session': False
            }
        }
        
        session_id = input_data['session']['session_id']
        if session_id not in app.config['users_state']:
            app.config['users_state'][session_id] = {
                'messages': []
            }
        
        user_state = app.config['users_state'][session_id]
        
        if input_data.get('request', {}).get('original_utterance'):
            user_message = clean_request(input_data['request']['original_utterance'])
            
            # Skip empty messages
            if not user_message:
                response['response']['text'] = 'Я вас не поняла. Пожалуйста, повторите вопрос.'
                response['response']['tts'] = 'Я вас не поняла. Пожалуйста, повторите вопрос.'
                return jsonify(response)
            
            user_state['messages'].append(user_message)
            
            # Choose AI provider based on configuration
            logger.info(f"Using AI provider: {AI_PROVIDER}")
            
            if AI_PROVIDER == 'gemini':
                bot_reply = ask_gemini(user_message, user_state['messages'])
            else:
                bot_reply = ask_openai(user_message, user_state['messages'])
                
            response['response']['text'] = bot_reply
            response['response']['tts'] = bot_reply + '<speaker audio="alice-sounds-things-door-2.opus">'
        else:
            response['response']['text'] = 'Я умный чат-бот. Спроси что-нибудь.'
            response['response']['tts'] = 'Я умный чат-бот. Спроси что-нибудь.'
        
        return jsonify(response)
    except Exception as e:
        logger.error(f"Request handling error: {str(e)}")
        return jsonify({
            "response": {
                "text": "Произошла ошибка при обработке запроса.",
                "end_session": False
            },
            "version": "1.0"
        })

@app.route('/', methods=['GET'])
def index():
    return "Сервис работает. Используйте метод POST для взаимодействия с API.", 405

if __name__ == '__main__':
    logger.info(f"Starting server with AI_PROVIDER={AI_PROVIDER}")
    app.run(host='0.0.0.0', port=int(os.getenv('PORT', '5000')))