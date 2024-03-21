from flask import Flask, request, jsonify
from groq import Groq
from langchain.chains import ConversationChain
from langchain.chains.conversation.memory import ConversationBufferWindowMemory
from langchain_groq import ChatGroq
from dotenv import load_dotenv
import os
from flask_cors import CORS
from flask_pymongo import PyMongo

load_dotenv()

app = Flask(__name__)
CORS(app)

# Configure MongoDB connection
app.config['MONGO_URI'] = os.environ.get('MONGO_URI')
mongo = PyMongo(app)

# Initialize Groq Langchain chat object
groq_api_key = os.environ['GROQ_API_KEY']
groq_chat = ChatGroq(
    groq_api_key=groq_api_key,
    model_name='mixtral-8x7b-32768',  # Default model
)

# Retrieve conversation history from MongoDB
def get_conversation_history():
    conversations = list(mongo.db.conversations.find({}, {'user_question': 1}))
    history = [conv['user_question'] for conv in conversations]
    return history

# Initialize conversation memory with retrieved history
conversation_history = get_conversation_history()
conversation_memory = ConversationBufferWindowMemory(k=5, initial_state=conversation_history)

# Initialize ConversationChain with Groq chat object and conversation memory
conversation = ConversationChain(llm=groq_chat, memory=conversation_memory)

@app.route('/')
def hello():
    return 'Hello, World!'

@app.route('/api/chat', methods=['POST'])
def chat():
    data = request.json
    user_question = data.get('userQuestion')
    model = data.get('model')
    
    # Change model if provided in the request
    if model and model in ['mixtral-8x7b-32768', 'llama2-70b-4096']:
        groq_chat.model_name = model

    # Save conversation to MongoDB
    conversation_entry = {
        'user_question': user_question,
        'response': conversation(user_question)
    }
    mongo.db.conversations.insert_one(conversation_entry)

    return jsonify(conversation_entry['response'])

if __name__ == '__main__':
    app.run()
