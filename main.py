from flask import Flask, render_template, request, Response, stream_with_context
import os
import time
import json
import google.generativeai as genai
from dotenv import load_dotenv
import markdown
import random
import requests
from bs4 import BeautifulSoup

app = Flask(__name__)

# Load environment variables
load_dotenv()

def load_system_instructions():
    with open('training.txt', 'r', encoding='utf-8') as file:
        return file.read()

# Configure the Gemini API
api_key = os.getenv("GEMINI_API_KEY")
if not api_key:
    raise ValueError("No GEMINI_API_KEY found in environment variables")

genai.configure(api_key=api_key)

generation_config = {
    "temperature": 1,
    "top_p": 0.95,
    "top_k": 64,
    "max_output_tokens": 8192,
}

safety_settings = [
    {"category": "HARM_CATEGORY_HARASSMENT", "threshold": "BLOCK_NONE"},
    {"category": "HARM_CATEGORY_HATE_SPEECH", "threshold": "BLOCK_NONE"},
    {"category": "HARM_CATEGORY_SEXUALLY_EXPLICIT", "threshold": "BLOCK_NONE"},
    {"category": "HARM_CATEGORY_DANGEROUS_CONTENT", "threshold": "BLOCK_NONE"},
]

model = genai.GenerativeModel(
    model_name="gemini-1.5-flash",
    safety_settings=safety_settings,
    generation_config=generation_config,
    system_instruction=load_system_instructions()
)

chat_sessions = {}

def process_markdown(text):
    html = markdown.markdown(text, extensions=['fenced_code', 'codehilite'])
    return html.replace('\n', '<br>')

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/init', methods=['GET'])
def init_chat():
    session_id = request.args.get('session_id')
    if not session_id:
        return Response("Session ID is required", status=400)

    if session_id not in chat_sessions:
        chat_sessions[session_id] = model.start_chat(history=[])

    initial_messages = [
        "Bruh, what do you want? 🙃",
        "Hey, hey! 🙌",
        "What is up! 🌟",
        "Mainframe AI reporting for duty! 🫡",
        "Mainframe AI on the beat ... What's good! 🎶",
        "Hello, fellow traveler! 🧙‍♂️"
        "I am an autonomous artificial intelligence from the planet of Cybertron, how can I help you today? 🤖",
        "Greetings, Earthling! What knowledge do you seek? 🚀",
        "Hey there! 👋",
        "Hey! Got questions? I’ve got answers! 📖"
        "Hello! What do you need help with? 🤗"
    ]

    random_initial_message = random.choice(initial_messages)

    return Response(json.dumps({"message": random_initial_message}),
                    content_type='application/json')

@app.route('/chat', methods=['GET', 'POST'])
def chat():
    session_id = request.args.get('session_id') or request.json.get('session_id')
    user_input = request.args.get('message') or request.json.get('message', '')

    if not session_id:
        return Response("Session ID is required", status=400)

    if session_id not in chat_sessions:
        chat_sessions[session_id] = model.start_chat(history=[])

    chat_session = chat_sessions[session_id]

    def generate_response():
        yield "data: " + json.dumps({"type": "start"}) + "\n\n"

        retries = 3
        for attempt in range(retries):
            try:
                response = chat_session.send_message(user_input, stream=True)

                full_response = ""
                for chunk in response:
                    full_response += chunk.text
                    yield "data: " + json.dumps({
                        "type": "chunk",
                        "content": chunk.text
                    }) + "\n\n"

                formatted_response = process_markdown(full_response)
                yield "data: " + json.dumps({
                    "type": "formatted",
                    "content": formatted_response
                }) + "\n\n"

                break

            except Exception as e:
                if "Resource has been exhausted" in str(e) and attempt < retries - 1:
                    time.sleep(2**attempt)
                    yield "data: " + json.dumps(
                        {
                            "type": "error",
                            "content": "Ugh, give me a sec... 🙄"
                        }) + "\n\n"
                else:
                    yield "data: " + json.dumps({
                        "type": "error",
                        "content": f"Oops! Looks like I short-circuited. Error: {str(e)} 🤖💥"
                    }) + "\n\n"
                    break

        yield "data: " + json.dumps({"type": "end"}) + "\n\n"

    return Response(stream_with_context(generate_response()),
                    content_type='text/event-stream')

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    app.run(host='0.0.0.0', port=port, debug=True)
