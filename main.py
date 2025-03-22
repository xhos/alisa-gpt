# Standard library imports
import os
import logging

# Third-party imports
from flask import Flask, request, jsonify
from dotenv import load_dotenv
from flask_session import Session
from google import genai
from google.genai import types
from openai import OpenAI

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

# Global configuration - choose AI provider based on available keys
openai_api_key = os.environ.get("OPENAI_API_KEY")
gemini_api_key = os.environ.get("GEMINI_API_KEY")

# Default to OpenAI if both are available, otherwise use whichever is available
if openai_api_key:
    AI_PROVIDER = os.getenv('AI_PROVIDER', 'openai').lower().strip()
    try:
        openai_client = OpenAI(api_key=openai_api_key)
        logger.info("OpenAI client initialized successfully")
    except Exception as e:
        logger.error(f"Failed to initialize OpenAI client: {str(e)}")
        if gemini_api_key:
            AI_PROVIDER = 'gemini'
        else:
            logger.error("No working AI providers found")
elif gemini_api_key:
    AI_PROVIDER = 'gemini'
    try:
        genai.configure(api_key=gemini_api_key)
        logger.info("Gemini API configured successfully")
    except Exception as e:
        logger.error(f"Failed to initialize Gemini: {str(e)}")
else:
    logger.error("No AI provider keys found. Application may not function correctly.")
    AI_PROVIDER = 'openai'  # Default even though it won't work

logger.info(f"AI_PROVIDER set to: {AI_PROVIDER}")

# Initialize session storage for users
if 'users_state' not in app.config:
    app.config['users_state'] = {}

# System prompt for AI assistants - consistent across providers
SYSTEM_PROMPT = """Ты голосовой ассистент, навык в умной колонке Яндекса. 
Пользователь может задать тебе односложный или более развернутый вопрос. 
Отвечай не слишком длинно, но и не чрезмерно кратко. 
API дает тебе лишь 3 секунды на ответ, это около параграфа. 
Твой ответ может включать в себя до трех предложений."""

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
    if not openai_api_key:
        logger.error("OpenAI API key not found")
        return 'Не удалось получить ответ от сервиса (ошибка конфигурации).'
    
    # Create the proper message format
    formatted_messages = [{"role": "system", "content": SYSTEM_PROMPT}]
    
    # Add conversation history
    for msg in messages[-10:]:  # Limit context window size
        formatted_messages.append({"role": "user", "content": msg})
    
    # Add the current message
    formatted_messages.append({"role": "user", "content": message})
    
    retries = 0
    while retries <= max_retries:
        try:
            # Use the official OpenAI client
            completion = openai_client.chat.completions.create(
                model="gpt-4o-mini",
                messages=formatted_messages,
                max_tokens=1000,
                temperature=0.7,
                timeout=30.0
            )
            
            logger.info("OpenAI API response received")
            
            # Extract the content from the response
            return completion.choices[0].message.content.strip()
            
        except Exception as e:
            retries += 1
            logger.error(f"OpenAI API error (attempt {retries}/{max_retries}): {str(e)}")
            
            if retries > max_retries:
                return 'Не удалось получить ответ от сервиса. Пожалуйста, попробуйте позже.'

# Function to interact with Gemini
def ask_gemini(message, messages, max_retries=2):
    if not gemini_api_key:
        logger.error("Gemini API key not found")
        return 'Не удалось получить ответ от сервиса (ошибка конфигурации).'
    
    retries = 0
    while retries <= max_retries:
        try:
            # Initialize the client with API key
            client = genai.Client(api_key=gemini_api_key)
            
            # Format previous messages
            previous_messages = ""
            if messages and len(messages) > 1:  # More than just the current message
                previous_messages = "\n".join([f"User: {msg}" for msg in messages[-5:-1]])
            
            # Create the content for the API request
            contents = [
                types.Content(
                    role="user",
                    parts=[types.Part.from_text(text=message)],
                )
            ]
            
            # If we have previous messages, add context
            if previous_messages:
                contents[0].parts.insert(
                    0, 
                    types.Part.from_text(text=f"Previous conversation:\n{previous_messages}")
                )
            
            # Configure the generation parameters
            generate_content_config = types.GenerateContentConfig(
                temperature=0.7,
                top_p=0.95,
                top_k=40,
                max_output_tokens=1000,
                response_mime_type="text/plain",
                system_instruction=[
                    types.Part.from_text(text=SYSTEM_PROMPT)
                ],
            )
            
            logger.info(f"Sending request to Gemini API with message: {message[:100]}...")
            
            # Generate content (non-streaming for simplicity)
            response = client.models.generate_content(
                model="gemini-2.0-flash-lite",
                contents=contents,
                config=generate_content_config,
            )
            
            logger.info("Gemini API response received")
            
            # Extract the response text
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
        'ai_provider': AI_PROVIDER,
        'openai_available': bool(openai_api_key),
        'gemini_available': bool(gemini_api_key)
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
            
            # Try preferred provider first, fall back to available provider if needed
            if AI_PROVIDER == 'gemini' and gemini_api_key:
                bot_reply = ask_gemini(user_message, user_state['messages'])
            elif openai_api_key:
                bot_reply = ask_openai(user_message, user_state['messages'])
            elif gemini_api_key:
                logger.warning("Falling back to Gemini as OpenAI is unavailable")
                bot_reply = ask_gemini(user_message, user_state['messages'])
            else:
                bot_reply = 'Извините, сервис временно недоступен.'
                
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