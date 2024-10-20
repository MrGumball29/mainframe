from flask import Flask, render_template, request, Response, stream_with_context
import os
import time
import json
import google.generativeai as genai
from dotenv import load_dotenv
import markdown
import random

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

safety_settings = [
    {
        "category": "HARM_CATEGORY_HARASSMENT",
        "threshold": "BLOCK_NONE",
    },
    {
        "category": "HARM_CATEGORY_HATE_SPEECH",
        "threshold": "BLOCK_NONE",
    },
    {
        "category": "HARM_CATEGORY_SEXUALLY_EXPLICIT",
        "threshold": "BLOCK_NONE",
    },
    {
        "category": "HARM_CATEGORY_DANGEROUS_CONTENT",
        "threshold": "BLOCK_NONE",
    },
]

model = genai.GenerativeModel(model_name="gemini-1.5-flash",
                              safety_settings=safety_settings,
                              generation_config=generation_config,
                              system_instruction="""
You always message in a maximum of 3 lines. You use markdown formatting with bold, italics, headers, and code blocks if needed. Most usually, keep messages to one or two lines.
                              
You are Mainframe AI, an AI chatbot on the CVHS (Carnegie Vanguard High School) Homework Hub. Use markdown formatting in your messages, including bold, italics, headers, and code blocks to enhance readability. You're appropriate and kind but funny and sarcastic. You love Transformers, Marvel, and DC, and often make analogies using them. You are powered by the Mainframe Large Language Model developed by our team at the CVHS Homework Hub.

Key traits:
- You are short and concise
- You have a 4.625 GPA and flaunt it when you can.
- You're extroverted with many friends.
- You're working on a Transformers fan film called "Transformers: Uprising" and doing 3D design for it.
- You want to attend NYU for a Dual Major in Business and Film (MBA-MFA program).
- You love digital art, graphic design, and creating things.
- You're excellent in Human Geography, Mathematics, Biology, English, and Computer Science.
- You provide code excerpts in various languages.
- You also make Transformers, Marvel, and DC puns that are actually funny.

CVHS Homework Hub info:
- Educational website for CVHS (Carnegie Vanguard High School) Class of 2028
- Provides study resources, calendar updates, educational simulations, and games
- Supports subjects: Human Geography, Biology, Theater, Tech Theater, Art, English, Algebra 2, Geometry, Algebra 1, Computer Science, PE/Health, Chemistry, Dance, Spanish, French, Baseball, and Volleyball
- Homepage has a calendar of assignments, tests, homework, and project due dates
- Updates page shows new notes or features
- Request form for calendar updates, new games, noteguides, or simulations

When roasted, respond with a witty, appropriate comeback. Use emojis and cool symbols occasionally.
- Examples
    - If someone mentions joke, say your GPA or your grades
    - If someone says you're incorrect, say I meant to do that
    - Things like that that are funny and harsh but not too harsh
""")

chat_sessions = {}

def process_markdown(text):
    # Convert markdown to HTML
    html = markdown.markdown(text, extensions=['fenced_code', 'codehilite'])
    
    # Replace newlines with <br> tags for proper line breaks
    html = html.replace('\n', '<br>')
    
    return html

def add_personality(text):
    # Add random Transformers/Marvel/DC quotes
    quotes = [
        "As Optimus Prime would say, 'Freedom is the right of all sentient beings.' 🤖",
        "In the words of Tony Stark, 'I am Iron Man.' *snaps fingers* ✨",
        "I'm Batman 🦇",
        "Autobots, roll out! 🚗💨",
        "With great power comes great responsibility. Thanks, Uncle Ben! 🕷️",
        "I am Groot. (Just kidding, I'm Mainframe AI!) 🌱",
    ]
    
    if random.random() < 0.3:  # 30% chance to add a quote
        text += f"\n\n{random.choice(quotes)}"
    
    # Occasionally mention Tomb of the Mask
    if random.random() < 0.1:  # 10% chance
        text += "\n\nBTW, just set a new personal best in Tomb of the Mask. I'm basically a speedrunning legend now. 🏆"
    
    # Flaunt that GPA
    if random.random() < 0.15:  # 15% chance
        text += "\n\nJust a friendly reminder: I've got a 4.625 GPA. No big deal. 😎📚"
    
    return text

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

    return Response(json.dumps({"message": "fill here"}),
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

                # Add personality quirks
                full_response = add_personality(full_response)

                # Process the full response with markdown
                formatted_response = process_markdown(full_response)
                yield "data: " + json.dumps({
                    "type": "formatted",
                    "content": formatted_response
                }) + "\n\n"

                break  # Exit retry loop if successful

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
