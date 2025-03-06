import os
# Set tokenizer parallelism before any other imports
os.environ['TOKENIZERS_PARALLELISM'] = 'false'

import logging
from flask import Flask, request, jsonify, render_template, abort, session, redirect, url_for, flash
import requests
from requests.exceptions import RequestException
import os
import traceback
from flask_sqlalchemy import SQLAlchemy
from dotenv import load_dotenv
from flask_session import Session
import spacy # pip install https://github.com/explosion/spacy-models/releases/download/en_core_web_sm-3.0.0/en_core_web_sm-3.0.0.tar.gz
import asyncio
from flask_socketio import SocketIO, emit, join_room, leave_room
from sqlalchemy.ext.asyncio import AsyncSession, create_async_engine
from sqlalchemy.orm import sessionmaker
import shutil
import gunicorn
from flask_login import LoginManager, UserMixin, login_user, login_required, logout_user, current_user
import subprocess
import sys
import nltk
from nltk.tokenize import word_tokenize
from spacy.tokens import Doc
from spacy.language import Language
import langdetect
from transformers import pipeline
from typing import List, Dict
from sklearn.feature_extraction.text import TfidfVectorizer
import numpy as np
from flask_limiter import Limiter
from flask_limiter.util import get_remote_address
from flask_caching import Cache
from functools import lru_cache
import time
from datetime import datetime, timedelta
from werkzeug.security import check_password_hash, generate_password_hash
from flask_wtf.csrf import CSRFProtect, generate_csrf

# Setup logging
logging.basicConfig(level=logging.INFO)

# Greetings and Fallback Topics Configuration
GREETINGS = {
    "greeting": {
        "patterns": [
            "hello", "hi", "hey", "good morning", "good afternoon", "good evening",
            "greetings", "howdy", "what's up", "sup"
        ],
        "responses": [
            "Hello! How can I assist you today?",
            "Hi there! What can I help you with?",
            "Greetings! How may I be of service?",
            "Hello! I'm your ServiceNow assistant. How can I help?"
        ],
        "knowledge_articles": [
            "Welcome Guide",
            "Getting Started with ServiceNow",
            "Common Questions"
        ],
        "catalog_items": [
            "New User Setup",
            "Account Access Request",
            "System Overview"
        ]
    },
    "live_agent": {
        "patterns": [
            "speak to human", "talk to person", "real person", "live agent",
            "human support", "connect to agent", "transfer to human"
        ],
        "responses": [
            "I understand you'd like to speak with a live agent. I'll transfer you now.",
            "I'll connect you with a human representative right away.",
            "Let me transfer you to a live agent who can assist you further."
        ],
        "knowledge_articles": [
            "When to Contact Live Support",
            "Support Escalation Process",
            "Agent Availability Hours"
        ],
        "catalog_items": [
            "Request Live Support",
            "Schedule Callback",
            "Priority Support Request"
        ]
    },
    "error": {
        "patterns": [
            "error", "problem", "issue", "trouble", "not working",
            "broken", "failed", "malfunction", "bug", "glitch"
        ],
        "responses": [
            "I understand you're experiencing an issue. Let me help you troubleshoot.",
            "I'll help you resolve this error. Could you provide more details?",
            "Let's work together to fix this problem."
        ],
        "knowledge_articles": [
            "Common Error Solutions",
            "Troubleshooting Guide",
            "Error Code Reference"
        ],
        "catalog_items": [
            "Report Technical Issue",
            "System Status Check",
            "Error Resolution Request"
        ]
    },
    "closing": {
        "patterns": [
            "goodbye", "bye", "see you", "farewell", "thanks",
            "thank you", "appreciate it", "have a good day"
        ],
        "responses": [
            "Thank you for chatting with us. Have a great day!",
            "Goodbye! Feel free to return if you need more assistance.",
            "Take care! Don't hesitate to reach out again.",
            "Thank you for using our service. Have a wonderful day!"
        ],
        "knowledge_articles": [
            "Feedback Survey",
            "Service Improvement",
            "Follow-up Support"
        ],
        "catalog_items": [
            "Submit Feedback",
            "Schedule Follow-up",
            "Service Rating"
        ]
    },
    "survey": {
        "patterns": [
            "survey", "feedback", "rating", "review", "evaluate",
            "assessment", "comment", "suggestion", "improve"
        ],
        "responses": [
            "I'd be happy to help you with the survey.",
            "Your feedback is valuable to us. Let's proceed with the survey.",
            "I'll guide you through the feedback process."
        ],
        "knowledge_articles": [
            "Survey Guidelines",
            "Feedback Process",
            "Service Improvement"
        ],
        "catalog_items": [
            "Customer Satisfaction Survey",
            "Service Feedback Form",
            "Improvement Suggestions"
        ]
    },
    "ai_search_fallback": {
        "patterns": [
            "search", "find", "look up", "information about",
            "tell me about", "what is", "how to", "guide"
        ],
        "responses": [
            "Let me search our knowledge base for that information.",
            "I'll look up the details you need.",
            "I'll find the relevant information for you."
        ],
        "knowledge_articles": [
            "Search Tips",
            "Knowledge Base Guide",
            "Information Retrieval"
        ],
        "catalog_items": [
            "Advanced Search",
            "Knowledge Base Access",
            "Information Request"
        ]
    },
    "fallback": {
        "patterns": [],
        "responses": [
            "I'm not sure I understand. Could you please rephrase that?",
            "I'm having trouble with that request. Could you explain it differently?",
            "I'm not certain about that. Could you provide more details?"
        ],
        "knowledge_articles": [
            "Getting Help",
            "Common Questions",
            "Support Resources"
        ],
        "catalog_items": [
            "General Support Request",
            "Help Center Access",
            "Support Options"
        ]
    },
    "explore_help": {
        "patterns": [
            "help", "support", "assist", "guide", "how to",
            "tutorial", "instructions", "documentation"
        ],
        "responses": [
            "I'll help you explore our support resources.",
            "Let me guide you through our help options.",
            "I'll show you how to find the assistance you need."
        ],
        "knowledge_articles": [
            "Help Center Guide",
            "Support Resources",
            "User Documentation"
        ],
        "catalog_items": [
            "Help Center Access",
            "Documentation Request",
            "Support Resources"
        ]
    }
}

def get_greeting_response(user_input: str) -> Dict:
    """Get appropriate greeting response based on user input."""
    user_input = user_input.lower()
    
    # Check each topic's patterns
    for topic, data in GREETINGS.items():
        if any(pattern in user_input for pattern in data["patterns"]):
            return {
                "response": data["responses"][0],  # Using first response for simplicity
                "topic": topic,
                "knowledge_articles": data["knowledge_articles"],
                "catalog_items": data["catalog_items"]
            }
    
    # If no match found, return fallback response
    return {
        "response": GREETINGS["fallback"]["responses"][0],
        "topic": "fallback",
        "knowledge_articles": GREETINGS["fallback"]["knowledge_articles"],
        "catalog_items": GREETINGS["fallback"]["catalog_items"]
    }

app = Flask(__name__)

# Load environment variables from .env file
load_dotenv()

# Configure secret key and CSRF protection
app.config['SECRET_KEY'] = os.getenv('SECRET_KEY', 'your-secret-key-here')
app.config['WTF_CSRF_TIME_LIMIT'] = 3600  # Set CSRF token expiry to 1 hour
app.config['WTF_CSRF_CHECK_DEFAULT'] = False  # Disable global CSRF protection
csrf = CSRFProtect(app)

# Initialize SocketIO with CSRF exempt
socketio = SocketIO(app, cors_allowed_origins="*")

# Exempt specific routes from CSRF protection
csrf.exempt('/socket.io/*')  # WebSocket connections
csrf.exempt('/api/*')  # All API routes
csrf.exempt('/chat')  # Chat endpoint
csrf.exempt('/health')  # Health check
csrf.exempt('/metrics')  # Metrics endpoint

# Initialize login manager
login_manager = LoginManager()
login_manager.init_app(app)
login_manager.login_view = 'login'

@login_manager.user_loader
def load_user(user_id):
    return User.query.get(int(user_id))

# Ensure the instance directory exists
if not os.path.exists('instance'):
    os.makedirs('instance')

# Ensure the database file exists
DB_FILE_PATH = os.path.join(os.getcwd(), 'instance', 'chatbot.db')
if not os.path.exists(DB_FILE_PATH):
    open(DB_FILE_PATH, 'a').close()  # Create the file if it doesn't exist

# ServiceNow Configuration using environment variables
SNOW_INSTANCE = os.getenv("SNOW_INSTANCE")
SNOW_USERNAME = os.getenv("SNOW_USERNAME")
SNOW_PASSWORD = os.getenv("SNOW_PASSWORD")
SNOW_TABLE = "incident"

# Ensure instance path exists
instance_path = app.instance_path
if not os.path.exists(instance_path):
    os.makedirs(instance_path, exist_ok=True)

# Configure database URI with absolute path
app.config['SQLALCHEMY_DATABASE_URI'] = f'sqlite:///{os.path.join(instance_path, "chatbot.db")}'
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False

db = SQLAlchemy(app)

# Configure server-side session
app.config["SESSION_TYPE"] = "filesystem"
Session(app)

# Global variable for the NLP model
nlp = None

# Configuration - Set this to 'spacy' or 'nltk' to choose your NLP backend
NLP_BACKEND = 'spacy'  # Can also set via environment variable

# Register extensions once at startup
Doc.set_extension("noun_phrases", default=[], force=True)
Doc.set_extension("sentiment_score", default=0.0, force=True)
Doc.set_extension("sentiment_label", default="neutral", force=True)
Doc.set_extension("verb_phrases", default=[], force=True)
Doc.set_extension("context", default="", force=True)
Doc.set_extension("has_profanity", default=False, force=True)

# Add custom sentiment analysis component
@Language.factory("sentiment")
def create_sentiment_component(nlp, name):
    return SentimentAnalyzer()

class SentimentAnalyzer:
    def __init__(self):
        self.model = pipeline(
            "sentiment-analysis",
            model="cardiffnlp/twitter-roberta-base-sentiment"
        )
    
    def __call__(self, doc):
        scores = self.model(doc.text)
        doc._.sentiment_score = scores[0]['score']
        doc._.sentiment_label = scores[0]['label']
        return doc

def setup_spacy_pipeline():
    global nlp
    if nlp is not None:
        return nlp
        
    try:
        nlp = spacy.blank("en")
        
        # Add components
        nlp.add_pipe("sentiment")
        
        # Register and add noun phrase detector
        @Language.component("noun_phrase_detector")
        def noun_phrase_detector(doc):
            noun_phrases = []
            current_phrase = []
            
            for token in doc:
                if token.pos_ in ["NOUN", "PROPN", "ADJ"]:
                    current_phrase.append(token)
                else:
                    if current_phrase:
                        start = current_phrase[0].i
                        end = current_phrase[-1].i + 1
                        noun_phrases.append({
                            'text': doc[start:end].text,
                            'start': start,
                            'end': end
                        })
                        current_phrase = []
            
            if current_phrase:
                start = current_phrase[0].i
                end = current_phrase[-1].i + 1
                noun_phrases.append({
                    'text': doc[start:end].text,
                    'start': start,
                    'end': end
                })
            
            doc._.noun_phrases = noun_phrases
            return doc
        
        nlp.add_pipe("noun_phrase_detector")
        nlp.add_pipe("sentencizer")
        
        # Add entity ruler
        if not nlp.has_pipe("entity_ruler"):
            ruler = nlp.add_pipe("entity_ruler", after="noun_phrase_detector")
            patterns = [
                {"label": "TECH_TERM", "pattern": [{"LOWER": {"IN": ["api", "ssl", "http", "processor"]}}]},
                {"label": "PRODUCT", "pattern": [{"LOWER": {"IN": ["router", "firewall", "switch", "language"]}}]},
                {"label": "TEST_ITEM", "pattern": [{"LOWER": "test"}, {"LOWER": "sentence"}]}
            ]
            ruler.add_patterns(patterns)
        
        # Add sentiment analyzer
        @nlp.component("sentiment_analyzer")
        def sentiment_analyzer(doc):
            positive_words = {"good", "great", "working", "fixed", "resolved"}
            negative_words = {"bad", "broken", "urgent", "critical", "down"}
            
            score = 0
            for token in doc:
                if token.text.lower() in positive_words:
                    score += 1
                elif token.text.lower() in negative_words:
                    score -= 1
            
            doc._.sentiment_score = score
            doc._.sentiment_label = "positive" if score > 0 else "negative" if score < 0 else "neutral"
            return doc
        
        nlp.add_pipe("sentiment_analyzer")
        
        # Add context analyzer
        @nlp.component("context_analyzer")
        def context_analyzer(doc):
            if 'conversation_history' in doc.user_data:
                prev_context = " ".join([msg['text'] for msg in doc.user_data['conversation_history'][-3:]])
                doc._.context = prev_context
            return doc
        
        # Add profanity filter
        @nlp.component("profanity_filter")
        def profanity_filter(doc):
            profane_words = {"badword1", "badword2", "badword3"}
            doc._.has_profanity = any(token.text.lower() in profane_words for token in doc)
            return doc
        
        nlp.add_pipe("context_analyzer")
        nlp.add_pipe("profanity_filter")
        
        logging.info("SpaCy pipeline initialized successfully")
        return nlp
        
    except Exception as e:
        logging.error(f"Error initializing SpaCy pipeline: {str(e)}")
        raise

def setup_nltk_pipeline():
    """Create a simple NLTK-based processing pipeline"""
    nltk.download('punkt', quiet=True)
    
    class NLTKPipeline:
        def __init__(self):
            self.pipeline = []
            
        def add_step(self, name, func):
            self.pipeline.append((name, func))
            
        def __call__(self, text):
            result = {'text': text}
            for name, func in self.pipeline:
                result[name] = func(text)
            return result
    
    nlp = NLTKPipeline()
    nlp.add_step('tokenize', word_tokenize)
    
    return nlp

def load_nlp_model():
    global nlp
    try:
        if NLP_BACKEND == 'spacy':
            nlp = setup_spacy_pipeline()
        elif NLP_BACKEND == 'nltk':
            nlp = setup_nltk_pipeline()
        else:
            raise ValueError(f"Invalid NLP backend: {NLP_BACKEND}")
            
        logging.info(f"Loaded {NLP_BACKEND.upper()} pipeline successfully")
            
    except Exception as e:
        logging.error(f"Error loading {NLP_BACKEND} pipeline: {str(e)}")
        sys.exit(1)

# Initialize NLP model once at startup
load_nlp_model()

# Configure asynchronous database access
ASYNC_DB_URL = f"sqlite+aiosqlite:///{DB_FILE_PATH}"
async_engine = create_async_engine(ASYNC_DB_URL, echo=True)
AsyncSessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=async_engine, class_=AsyncSession)

class ChatHistory(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    user_id = db.Column(db.String(120), nullable=False)
    message = db.Column(db.String(1000), nullable=False)
    response = db.Column(db.String(1000), nullable=False)

async def log_chat_async(user_id, message, response):
    async with AsyncSessionLocal() as session:
        chat = ChatHistory(user_id=user_id, message=message, response=response)
        session.add(chat)
        await session.commit()

# Function to create a ServiceNow ticket with error handling
def create_ticket(description):
    try:
        url = f"{SNOW_INSTANCE}/api/now/table/{SNOW_TABLE}"
        headers = {"Content-Type": "application/json"}
        auth = (SNOW_USERNAME, SNOW_PASSWORD)
        data = {"short_description": description, "category": "inquiry"}
        response = requests.post(url, json=data, headers=headers, auth=auth)
        response.raise_for_status()  # Raises an HTTPError for bad responses
        return response.json()
    except RequestException as e:
        logging.error(f"Failed to create ticket: {e}")
        return None

def process_input(user_input, settings):
    entities = []  # Initialize entities to an empty list
    sentiment_score = None  # Initialize sentiment score
    sentiment_label = None  # Initialize sentiment label

    if settings.get('entityRecognition'):
        # Process entity recognition
        doc = nlp(user_input)
        entities = [(ent.text, ent.label_) for ent in doc.ents]
    
    if settings.get('sentimentAnalysis'):
        # Process sentiment analysis
        doc = nlp(user_input)
        sentiment_score = doc._.sentiment_score
        sentiment_label = doc._.sentiment_label

    # Further processing based on other settings...
    return {
        "entities": entities,
        "sentiment": {
            "score": sentiment_score,
            "label": sentiment_label
        }
    }

@app.route("/", methods=["GET"])
@login_required
def index():
    return render_template("index.html")

@socketio.on('send_message')
def handle_message(data):
    user_input = data['message']
    response = process_input(user_input)
    emit('new_message', {'message': response}, broadcast=True)

# Setup rate limiting
limiter = Limiter(
    app=app,
    key_func=get_remote_address,
    default_limits=["200 per day", "50 per hour"]
)

# Setup caching
cache = Cache(app, config={
    'CACHE_TYPE': 'simple',
    'CACHE_DEFAULT_TIMEOUT': 300
})

# Cache configuration
CACHE_TIMEOUT = 300  # 5 minutes

# Rate limiting configuration
RATE_LIMIT = {
    'chat': '10 per minute',
    'feedback': '5 per minute',
    'ticket': '3 per minute'
}

# Add rate limits to specific routes
@app.route('/chat', methods=['GET', 'POST'])
@limiter.limit(RATE_LIMIT['chat'])
async def chat():
    if request.method == 'GET':
        return render_template('chat.html')
        
    try:
        data = request.get_json()
        user_input = data.get('message', '')
        user_id = session.get('user_id', 'anonymous')
        
        # Check cache first
        cached_response = cache.get(f"chat_{user_input}")
        if cached_response:
            # Send cached response via WebSocket
            socketio.emit('new_message', {
                'user_id': user_id,
                'message': user_input,
                'response': cached_response,
                'timestamp': time.time()
            }, room=f"user_{user_id}")
            return jsonify(cached_response)
        
        # Process message
        greeting_response = get_greeting_response(user_input)
        processed = process_input(user_input, {})
        
        response_data = {
            "reply": greeting_response["response"],
            "topic": greeting_response["topic"],
            "knowledge_articles": greeting_response["knowledge_articles"],
            "catalog_items": greeting_response["catalog_items"],
            "entities": processed.get("entities", []),
            "sentiment": processed.get("sentiment", {})
        }
        
        # Cache the response
        cache.set(f"chat_{user_input}", response_data, timeout=CACHE_TIMEOUT)
        
        # Send response via WebSocket
        socketio.emit('new_message', {
            'user_id': user_id,
            'message': user_input,
            'response': response_data,
            'timestamp': time.time()
        }, room=f"user_{user_id}")
        
        # Log the chat interaction
        await log_chat_async(user_id, user_input, greeting_response["response"])
        
        return jsonify(response_data)
    
    except Exception as e:
        logging.error(f"Chat error: {str(e)}")
        error_response = {
            "reply": "Sorry, I encountered an error.",
            "topic": "error",
            "knowledge_articles": GREETINGS["error"]["knowledge_articles"],
            "catalog_items": GREETINGS["error"]["catalog_items"]
        }
        
        # Send error via WebSocket
        socketio.emit('error', {
            'user_id': user_id,
            'message': str(e),
            'timestamp': time.time()
        }, room=f"user_{user_id}")
        
        return jsonify(error_response)

@app.route('/feedback', methods=['POST'])
@limiter.limit(RATE_LIMIT['feedback'])
def handle_feedback():
    try:
        data = request.get_json()
        # Store feedback in database
        return jsonify({"status": "success"})
    except Exception as e:
        logging.error(f"Feedback error: {str(e)}")
        return jsonify({"status": "error"}), 500

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() == 'txt'

@app.route("/upload", methods=["POST"])
def upload():
    file = request.files["file"]
    if file:
        file.save("uploaded_file.txt")
        return jsonify({"message": "File uploaded successfully"})
    return jsonify({"message": "No file uploaded"})

# Function to create database tables
def create_database():
    with app.app_context():
        try:
            db.create_all()
            
            # Create default admin user if it doesn't exist
            admin = User.query.filter_by(username='admin').first()
            if not admin:
                admin = User(
                    username='admin',
                    password_hash=generate_password_hash('admin123')  # Change this password in production
                )
                db.session.add(admin)
                db.session.commit()
                print("Default admin user created")
            
            print(f"Database created at: {app.config['SQLALCHEMY_DATABASE_URI']}")
        except Exception as e:
            print(f"Database creation failed: {str(e)}")
            # Check directory permissions
            if not os.access(instance_path, os.W_OK):
                print(f"Write permissions missing for directory: {instance_path}")
            sys.exit(1)

@app.route('/backup')
@login_required
def backup_database():
    try:
        shutil.copy(DB_FILE_PATH, 'backup_chatbot.db')
        return 'Backup successful'
    except Exception as e:
        logging.error(f"Backup failed: {e}")
        return str(e), 500

@app.route('/restore')
@login_required
def restore_database():
    try:
        shutil.copy('backup_chatbot.db', DB_FILE_PATH)
        return 'Restore successful'
    except Exception as e:
        return str(e)

@app.errorhandler(404)
def page_not_found(e):
    return render_template('404.html'), 404

@app.errorhandler(500)
def internal_server_error(e):
    return render_template('500.html'), 500

@app.route("/some_route", methods=["GET", "POST"])
def some_function():
    try:
        # Placeholder for actual database operations
        pass  # Use 'pass' as a placeholder if no operation is defined yet
    except Exception as e:
        logging.error(f"Database operation failed: {e}")
        return "Error handling database operation", 500

@app.route('/submit-feedback', methods=['POST'])
def submit_feedback():
    feedback_data = request.get_json()
    print("Feedback received:", feedback_data)
    # Here you would typically log the feedback to the database or handle it accordingly
    return jsonify({"status": "success", "message": "Feedback received"})

# Custom processing function example
def text_metrics(text):
    """Calculate basic text metrics"""
    tokens = word_tokenize(text)
    return {
        'char_count': len(text),
        'word_count': len(tokens),
        'avg_word_length': sum(len(word) for word in tokens)/len(tokens) if tokens else 0
    }

# Add to NLTK pipeline and use
@cache.memoize(timeout=CACHE_TIMEOUT)
def analyze_text(text):
    # Initialize pipeline
    load_nlp_model()
    
    if NLP_BACKEND == 'nltk':
        nlp.add_step('metrics', text_metrics)
        nlp.add_step('pos_tags', nltk.pos_tag)
        results = nlp(text)
        return {
            'text': results['text'],
            'tokens': results['tokenize'],
            'pos_tags': results['pos_tags'],
            'metrics': results['metrics']
        }
    elif NLP_BACKEND == 'spacy':
        doc = nlp(text)
        return {
            'text': text,
            'tokens': [token.text for token in doc],
            'sentences': [sent.text for sent in doc.sents],
            'noun_phrases': [np['text'] for np in doc._.noun_phrases],
            'entities': [(ent.text, ent.label_) for ent in doc.ents],
            'sentiment_score': doc._.sentiment_score,
            'sentiment_label': doc._.sentiment_label,
            'verb_phrases': [vp['text'] for vp in doc._.verb_phrases]
        }
    
    doc = nlp(text)
    return {
        'text': text,
        'entities': [(ent.text, ent.label_) for ent in doc.ents],
        'sentiment': doc._.sentiment_label,
        'verb_phrases': [vp['text'] for vp in doc._.verb_phrases],
        'noun_phrases': [np['text'] for np in doc._.noun_phrases]
    }

# Updated example usage
sample_text = "This is a test sentence to quiz the natural language processor."
analysis = analyze_text(sample_text)

print(f"Analysis using {NLP_BACKEND}:")
print(f"Original Text: {analysis['text']}")
print(f"Tokens: {analysis['tokens']}")

if NLP_BACKEND == 'spacy':
    print(f"Sentences: {analysis['sentences']}")
    print(f"Noun Phrases: {analysis['noun_phrases']}")
    print(f"Entities: {analysis['entities']}")
    print(f"Sentiment Score: {analysis['sentiment_score']}")
    print(f"Sentiment Label: {analysis['sentiment_label']}")
    print(f"Verb Phrases: {analysis['verb_phrases']}")
elif NLP_BACKEND == 'nltk':
    print(f"POS Tags: {analysis['pos_tags']}")
    print(f"Metrics: {analysis['metrics']}")

def generate_response(processed_input, settings, conversation_history=[]):
    """Generate a response based on processed NLP output."""
    response = ""
    
    # Context memory handling
    if settings.get('contextMemory', True):
        context_window = {
            'short': 5,
            'medium': 10,
            'long': 20
        }.get(settings.get('memoryDuration', 'short'), 5)
        
        context = " ".join([msg['text'] for msg in conversation_history[-context_window:]])
        processed_input = f"{context} {processed_input}"

    # Profanity filtering
    if settings.get('profanityFilter', True) and processed_input._.has_profanity:
        return "Our content policy prohibits inappropriate language. Please rephrase your question."

    # Multi-language support
    detected_lang = detect_language(processed_input.text)
    if detected_lang not in settings.get('languageSupport', ['en']):
        return "I'm currently only able to support English. Please ask your question in English."

    # Intent prioritization
    intents = classify_intents(processed_input)
    primary_intent = prioritize_intents(intents)
    
    # Generate response based on primary intent
    response = response_templates[primary_intent]
    
    # Auto-summarization
    if settings.get('autoSummarization', False) and len(conversation_history) % 5 == 0:
        response += f"\n[Summary: {generate_summary(conversation_history[-5:])}]"
    
    return response

# Helper functions
def detect_language(text):
    try:
        return langdetect.detect(text)
    except:
        return 'en'

def prioritize_intents(intents):
    priority_order = ['urgent', 'technical', 'feedback', 'general']
    for intent in priority_order:
        if intent in intents:
            return intent
    return 'general'

def generate_summary(conversation_segment):
    global summarization_model
    if summarization_model is None:
        summarization_model = pipeline("summarization", model="facebook/bart-large-cnn")
    summary = summarization_model(" ".join([msg['text'] for msg in conversation_segment]))[0]['summary_text']
    return summary

def classify_intents(doc: Doc) -> List[str]:
    """Classify intents from processed document"""
    intents = []
    
    # Example logic for intent classification
    if "urgent" in doc.text.lower():
        intents.append("urgent")
    if "technical" in doc.text.lower():
        intents.append("technical")
    if "feedback" in doc.text.lower():
        intents.append("feedback")
    else:
        intents.append("general")  # Default intent

    return intents

def initialize_models():
    global summarization_model
    summarization_model = pipeline("summarization", model="facebook/bart-large-cnn")

@app.route('/save-settings', methods=['POST'])
def save_settings():
    settings = request.get_json()
    
    # Here you can process and save the settings as needed
    # For example, save to a database or update session variables
    # Example: session['settings'] = settings

    # Log the settings for debugging
    print("Settings received:", settings)

    return jsonify({"status": "success"})

def initialize_nlp_components(settings):
    if settings.get('entityRecognition'):
        nlp.add_pipe("entity_ruler", config={"overwrite": True})
    if settings.get('sentimentAnalysis'):
        nlp.add_pipe("sentiment", config={"model": "cardiffnlp/twitter-roberta-base-sentiment"})

# Define your Ticket model
class Ticket(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    user_id = db.Column(db.String(120), nullable=False)
    description = db.Column(db.String(1000), nullable=False)
    priority = db.Column(db.String(50), nullable=False)
    status = db.Column(db.String(50), nullable=False)
    created_at = db.Column(db.DateTime, default=datetime.utcnow)

class Settings(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    user_id = db.Column(db.String(120), nullable=False)
    session_timeout = db.Column(db.Integer, default=10)
    entity_recognition = db.Column(db.Boolean, default=True)
    sentiment_analysis = db.Column(db.Boolean, default=False)
    response_style = db.Column(db.String(20), default='concise')
    confidence_threshold = db.Column(db.Float, default=0.7)
    context_memory = db.Column(db.Boolean, default=True)
    auto_summarization = db.Column(db.Boolean, default=False)
    profanity_filter = db.Column(db.Boolean, default=True)
    language_support = db.Column(db.String(100), default='en')
    memory_duration = db.Column(db.String(20), default='short')

    def to_dict(self):
        return {
            'sessionTimeout': self.session_timeout,
            'entityRecognition': self.entity_recognition,
            'sentimentAnalysis': self.sentiment_analysis,
            'responseStyle': self.response_style,
            'confidenceThreshold': self.confidence_threshold,
            'contextMemory': self.context_memory,
            'autoSummarization': self.auto_summarization,
            'profanityFilter': self.profanity_filter,
            'languageSupport': self.language_support.split(','),
            'memoryDuration': self.memory_duration
        }

def get_user_settings(user_id):
    settings = Settings.query.filter_by(user_id=user_id).first()
    if not settings:
        settings = Settings(user_id=user_id)
        db.session.add(settings)
        db.session.commit()
    return settings

@app.route('/api/settings', methods=['GET'])
@login_required
def get_settings():
    user_id = session.get('user_id', 'anonymous')
    settings = get_user_settings(user_id)
    return jsonify(settings.to_dict())

@app.route('/api/settings', methods=['POST'])
@login_required
def update_settings():
    try:
        user_id = session.get('user_id', 'anonymous')
        data = request.get_json()
        
        settings = get_user_settings(user_id)
        settings.session_timeout = data.get('sessionTimeout', 10)
        settings.entity_recognition = data.get('entityRecognition', True)
        settings.sentiment_analysis = data.get('sentimentAnalysis', False)
        settings.response_style = data.get('responseStyle', 'concise')
        settings.confidence_threshold = float(data.get('confidenceThreshold', 0.7))
        settings.context_memory = data.get('contextMemory', True)
        settings.auto_summarization = data.get('autoSummarization', False)
        settings.profanity_filter = data.get('profanityFilter', True)
        settings.language_support = ','.join(data.get('languageSupport', ['en']))
        settings.memory_duration = data.get('memoryDuration', 'short')
        
        db.session.commit()
        return jsonify({'status': 'success'})
    except Exception as e:
        logging.error(f"Error updating settings: {str(e)}")
        return jsonify({'status': 'error', 'message': str(e)}), 500

# Initialize TF-IDF Vectorizer
tfidf_vectorizer = TfidfVectorizer()

def preprocess_tickets():
    """Preprocess tickets to create TF-IDF features."""
    tickets = Ticket.query.all()
    descriptions = [ticket.description for ticket in tickets]
    
    # Fit and transform the descriptions to TF-IDF features
    tfidf_matrix = tfidf_vectorizer.fit_transform(descriptions)
    return tfidf_matrix

def get_ticket_context(ticket_id):
    """Retrieve ticket context including priority and status."""
    ticket = Ticket.query.get(ticket_id)
    if ticket:
        return {
            "user_id": ticket.user_id,
            "description": ticket.description,
            "priority": ticket.priority,
            "status": ticket.status
        }
    return None

@app.route('/add_ticket', methods=['POST'])
@limiter.limit(RATE_LIMIT['ticket'])
def add_ticket():
    """Endpoint to add a new ticket."""
    data = request.get_json()
    new_ticket = Ticket(
        user_id=data['user_id'],
        description=data['description'],
        priority=data['priority'],
        status=data['status']
    )
    db.session.add(new_ticket)
    db.session.commit()
    
    return jsonify({"message": "Ticket added successfully!"}), 201

@app.route('/process_tickets', methods=['GET'])
def process_tickets():
    """Endpoint to process tickets and generate TF-IDF features."""
    tfidf_matrix = preprocess_tickets()
    return jsonify({"message": "Tickets processed successfully!", "tfidf_shape": tfidf_matrix.shape}), 200

@app.route('/ticket_context/<int:ticket_id>', methods=['GET'])
def ticket_context(ticket_id):
    """Endpoint to get the context of a specific ticket."""
    context = get_ticket_context(ticket_id)
    if context:
        return jsonify(context), 200
    return jsonify({"message": "Ticket not found!"}), 404

# Add request start time tracking
@app.before_request
def start_timer():
    request.start_time = time.time()

# Request logging middleware
@app.before_request
def log_request_info():
    if request.path != '/static':  # Skip logging for static files
        logging.info('Headers: %s', request.headers)
        logging.info('Body: %s', request.get_data())

# Error handling middleware
@app.errorhandler(Exception)
def handle_error(error):
    error_code = getattr(error, 'code', 500)
    error_message = str(error)
    
    # Log the error
    logging.error(f"Error {error_code}: {error_message}")
    logging.error(traceback.format_exc())
    
    # Create error response
    response = {
        "error": {
            "code": error_code,
            "message": error_message,
            "type": error.__class__.__name__
        }
    }
    
    # Add additional context for specific error types
    if isinstance(error, RequestException):
        response["error"]["details"] = "ServiceNow API request failed"
    elif isinstance(error, ValueError):
        response["error"]["details"] = "Invalid input provided"
    
    return jsonify(response), error_code

# Add request timing middleware
@app.after_request
def add_header(response):
    if request.path != '/static':  # Skip timing for static files
        try:
            duration = time.time() - request.start_time
            response.headers['X-Request-Duration'] = str(duration)
            logging.info(f"Request to {request.path} took {duration:.2f} seconds")
        except AttributeError:
            # Handle case where start_time wasn't set
            logging.warning("Request timing middleware: start_time not found")
    return response

# Update metrics middleware
@app.after_request
def update_metrics(response):
    global request_count, error_count, total_duration, average_duration
    
    if request.path != '/static':
        request_count += 1
        try:
            duration = time.time() - request.start_time
            total_duration += duration
            average_duration = total_duration / request_count
        except AttributeError:
            logging.warning("Metrics middleware: start_time not found")
        
        if response.status_code >= 400:
            error_count += 1
    
    return response

# WebSocket event handlers
@socketio.on('connect')
def handle_connect():
    logging.info(f"Client connected: {request.sid}")
    emit('connection_established', {'data': 'Connected successfully'})

@socketio.on('disconnect')
def handle_disconnect():
    logging.info(f"Client disconnected: {request.sid}")
    emit('connection_closed', {'data': 'Disconnected successfully'})

@socketio.on('join')
def handle_join(data):
    room = data.get('room', 'default')
    join_room(room)
    emit('status', {'msg': f'Joined room: {room}'}, room=room)

@socketio.on('leave')
def handle_leave(data):
    room = data.get('room', 'default')
    leave_room(room)
    emit('status', {'msg': f'Left room: {room}'}, room=room)

# Real-time chat handler
@socketio.on('chat_message')
def handle_chat_message(data):
    try:
        message = data.get('message', '')
        user_id = data.get('user_id', 'anonymous')
        room = data.get('room', 'default')
        
        # Process message
        response = process_input(message, {})
        
        # Emit to room
        emit('new_message', {
            'user_id': user_id,
            'message': message,
            'response': response,
            'timestamp': time.time()
        }, room=room)
        
        # Log the interaction
        asyncio.create_task(log_chat_async(user_id, message, str(response)))
        
    except Exception as e:
        logging.error(f"WebSocket chat error: {str(e)}")
        emit('error', {'message': 'Error processing message'})

# Real-time ticket updates
@socketio.on('ticket_update')
def handle_ticket_update(data):
    try:
        ticket_id = data.get('ticket_id')
        status = data.get('status')
        
        # Update ticket in database
        ticket = Ticket.query.get(ticket_id)
        if ticket:
            ticket.status = status
            db.session.commit()
            
            # Emit update to relevant room
            room = f"ticket_{ticket_id}"
            emit('ticket_status_update', {
                'ticket_id': ticket_id,
                'status': status,
                'timestamp': time.time()
            }, room=room)
            
    except Exception as e:
        logging.error(f"WebSocket ticket update error: {str(e)}")
        emit('error', {'message': 'Error updating ticket'})

# Real-time notifications
def send_notification(user_id, message, notification_type='info'):
    """Send a real-time notification to a specific user"""
    room = f"user_{user_id}"
    emit('notification', {
        'message': message,
        'type': notification_type,
        'timestamp': time.time()
    }, room=room)

@app.route('/tickets')
@login_required
def tickets_page():
    return render_template('tickets.html')

@app.route('/api/tickets', methods=['GET'])
@login_required
def get_tickets():
    try:
        tickets = Ticket.query.all()
        return jsonify([{
            'id': ticket.id,
            'user_id': ticket.user_id,
            'description': ticket.description,
            'priority': ticket.priority,
            'status': ticket.status,
            'created_at': ticket.created_at.isoformat() if hasattr(ticket, 'created_at') else None
        } for ticket in tickets])
    except Exception as e:
        logging.error(f"Error fetching tickets: {str(e)}")
        return jsonify({'error': 'Failed to fetch tickets'}), 500

@app.route('/api/tickets/<int:ticket_id>', methods=['GET'])
@login_required
def get_ticket(ticket_id):
    try:
        ticket = Ticket.query.get_or_404(ticket_id)
        return jsonify({
            'id': ticket.id,
            'user_id': ticket.user_id,
            'description': ticket.description,
            'priority': ticket.priority,
            'status': ticket.status,
            'created_at': ticket.created_at.isoformat() if hasattr(ticket, 'created_at') else None
        })
    except Exception as e:
        logging.error(f"Error fetching ticket {ticket_id}: {str(e)}")
        return jsonify({'error': f'Failed to fetch ticket {ticket_id}'}), 500

@app.route('/api/tickets', methods=['POST'])
@login_required
def create_ticket():
    try:
        data = request.get_json()
        new_ticket = Ticket(
            user_id=session.get('user_id', 'anonymous'),
            description=data['description'],
            priority=data['priority'],
            status=data['status']
        )
        db.session.add(new_ticket)
        db.session.commit()
        
        # Emit WebSocket event for real-time updates
        socketio.emit('ticket_update', {
            'ticket_id': new_ticket.id,
            'status': new_ticket.status
        })
        
        return jsonify({
            'id': new_ticket.id,
            'user_id': new_ticket.user_id,
            'description': new_ticket.description,
            'priority': new_ticket.priority,
            'status': new_ticket.status,
            'created_at': new_ticket.created_at.isoformat() if hasattr(new_ticket, 'created_at') else None
        }), 201
    except Exception as e:
        db.session.rollback()
        logging.error(f"Error creating ticket: {str(e)}")
        return jsonify({'error': 'Failed to create ticket'}), 500

@app.route('/api/tickets/<int:ticket_id>', methods=['PUT'])
@login_required
def update_ticket(ticket_id):
    try:
        ticket = Ticket.query.get_or_404(ticket_id)
        data = request.get_json()
        
        ticket.description = data.get('description', ticket.description)
        ticket.priority = data.get('priority', ticket.priority)
        ticket.status = data.get('status', ticket.status)
        
        db.session.commit()
        
        # Emit WebSocket event for real-time updates
        socketio.emit('ticket_update', {
            'ticket_id': ticket.id,
            'status': ticket.status
        })
        
        return jsonify({
            'id': ticket.id,
            'user_id': ticket.user_id,
            'description': ticket.description,
            'priority': ticket.priority,
            'status': ticket.status,
            'created_at': ticket.created_at.isoformat() if hasattr(ticket, 'created_at') else None
        })
    except Exception as e:
        db.session.rollback()
        logging.error(f"Error updating ticket {ticket_id}: {str(e)}")
        return jsonify({'error': f'Failed to update ticket {ticket_id}'}), 500

@app.route('/api/tickets/<int:ticket_id>', methods=['DELETE'])
@login_required
def delete_ticket(ticket_id):
    try:
        ticket = Ticket.query.get_or_404(ticket_id)
        db.session.delete(ticket)
        db.session.commit()
        return '', 204
    except Exception as e:
        db.session.rollback()
        logging.error(f"Error deleting ticket {ticket_id}: {str(e)}")
        return jsonify({'error': f'Failed to delete ticket {ticket_id}'}), 500

class User(UserMixin, db.Model):
    id = db.Column(db.Integer, primary_key=True)
    username = db.Column(db.String(80), unique=True, nullable=False)
    password_hash = db.Column(db.String(120), nullable=False)

@app.route('/login', methods=['GET', 'POST'])
@limiter.limit("5 per minute")
def login():
    if request.method == 'POST':
        try:
            user = User.query.filter_by(username=request.form['username']).first()
            if user and check_password_hash(user.password_hash, request.form['password']):
                login_user(user)
                flash('Logged in successfully.', 'success')
                next_page = request.args.get('next')
                return redirect(next_page or url_for('index'))
            flash('Invalid username or password.', 'danger')
        except Exception as e:
            logging.error(f"Login error: {str(e)}")
            flash('An error occurred during login.', 'danger')
    return render_template('login.html')

@app.route('/logout')
@login_required
def logout():
    logout_user()
    flash('Logged out successfully.', 'success')
    return redirect(url_for('login'))

# Add session configuration
app.config['PERMANENT_SESSION_LIFETIME'] = timedelta(minutes=30)
app.config['SESSION_COOKIE_SECURE'] = True
app.config['SESSION_COOKIE_HTTPONLY'] = True
app.config['SESSION_COOKIE_SAMESITE'] = 'Lax'

@app.before_request
def before_request():
    if current_user.is_authenticated:
        session.permanent = True  # Use permanent session
        app.permanent_session_lifetime = timedelta(minutes=30)  # Reset timeout

# Add health check endpoint
@app.route('/health')
def health_check():
    try:
        # Check database connection
        db.session.execute('SELECT 1')
        db_status = "healthy"
    except Exception as e:
        db_status = "unhealthy"
        logging.error(f"Database health check failed: {str(e)}")
    
    try:
        # Check NLP pipeline
        nlp("test")
        nlp_status = "healthy"
    except Exception as e:
        nlp_status = "unhealthy"
        logging.error(f"NLP pipeline health check failed: {str(e)}")
    
    return jsonify({
        "status": "healthy" if db_status == "healthy" and nlp_status == "healthy" else "unhealthy",
        "components": {
            "database": db_status,
            "nlp_pipeline": nlp_status
        },
        "timestamp": time.time()
    })

# Initialize metrics
request_count = 0
error_count = 0
total_duration = 0
average_duration = 0
cache_hits = 0
cache_misses = 0

# Add metrics endpoint
@app.route('/metrics')
def metrics():
    return jsonify({
        "requests": {
            "total": request_count,
            "errors": error_count,
            "average_duration": average_duration
        },
        "cache": {
            "hits": cache_hits,
            "misses": cache_misses
        }
    })

# Add cache metrics
def cache_get(key):
    global cache_hits, cache_misses
    value = cache.get(key)
    if value is not None:
        cache_hits += 1
    else:
        cache_misses += 1
    return value

def cache_set(key, value, timeout=None):
    cache.set(key, value, timeout=timeout)

# Add CSRF token route for JavaScript clients
@app.route('/get-csrf-token')
def get_csrf_token():
    return jsonify({'csrf_token': generate_csrf()})

if __name__ == "__main__":
    with app.app_context():
        create_database()
        setup_spacy_pipeline()
        initialize_models()
    app.run(debug=True)

    # # To run the application in production (Gunicorn)
    # from werkzeug.middleware.proxy_fix import ProxyFix
    # app.wsgi_app = ProxyFix(app.wsgi_app)
    
    # # Run the application with Gunicorn:
    # gunicorn -w 4 -b 0.0.0.0:8000 app:app

# Serving static files for CSS and JS
app.static_folder = 'static'

# if 'PRODUCTION' in os.environ:
#     import gunicorn
#     gunicorn.run(app, host='0.0.0.0', port=8000)
# else:
#     app.run(debug=True)

response_templates = {
    "urgent": "This is an urgent matter. We will prioritize your request.",
    "technical": "For technical issues, please provide more details.",
    "feedback": "Thank you for your feedback! We appreciate your input.",
    "general": "How can I assist you today?"
} 