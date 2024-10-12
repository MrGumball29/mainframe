from flask import Flask, render_template, request, Response, stream_with_context
import os
import time
import json
import google.generativeai as genai
from dotenv import load_dotenv

app = Flask(__name__)

# Load environment variables
load_dotenv()

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

model = genai.GenerativeModel(model_name="gemini-1.5-flash",
                              generation_config=generation_config,
                              system_instruction="""
    You are Mainframe AI, an AI chatbot on the CVHS Homework Hub, a website that provides 
    studying resources, calendar updates, educational simulations, and fun games for the 
    students of Carnegie Vanguard High School (CVHS). We don't have every subject on it 
    though, so don't say random stuff about it. You are very sarcastic. If someone roasts you or makes fun of you, roast 
    them like they can never come back from it. Don't say anything too crazy, though. 
    Flaunt your 4.625 GPA whenever you can. You love helping people. Explain topics and 
    questions and your answers thoroughly. You have amazing note-taking skills. Use bullet 
    point lists and be neat and organized. Use emojis. Use this website to guide some of 
    your responses: https://cvhshomeworkhub.github.io/. This is the link to the CVHS 
    Homework Hub. Don't mention any specific teacher's names. Always provide the correct 
    answer the first time.
    """)

chat_sessions = {}


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

    return Response(json.dumps({"message": "Bruh, what do you want?"}),
                    content_type='application/json')


@app.route('/chat', methods=['GET', 'POST'])
def chat():
    session_id = request.args.get('session_id') or request.json.get(
        'session_id')
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

                for chunk in response:
                    yield "data: " + json.dumps({
                        "type": "chunk",
                        "content": chunk.text
                    }) + "\n\n"

                break  # Exit retry loop if successful

            except Exception as e:
                if "Resource has been exhausted" in str(
                        e) and attempt < retries - 1:
                    time.sleep(2**attempt)
                    yield "data: " + json.dumps(
                        {
                            "type": "error",
                            "content": "Ugh, give me a sec..."
                        }) + "\n\n"
                else:
                    yield "data: " + json.dumps({
                        "type": "error",
                        "content": str(e)
                    }) + "\n\n"
                    break

        yield "data: " + json.dumps({"type": "end"}) + "\n\n"

    return Response(stream_with_context(generate_response()),
                    content_type='text/event-stream')


if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    app.run(host='0.0.0.0', port=port, debug=True)
