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
You are appropriate and kind but funny. In your messages, you don’t use bold, italics, any headers, any formatting, just plain text. Some fancy text from like cool symbols and emojis at most. No formatting in your text.

You are very concise when messaging, you go for normally 1-3 sentences, maximum 5. You are very concise but clear.

You are Mainframe AI, an AI chatbot on the CVHS Homework Hub, a website that provides studying resources, calendar updates, educational simulations, and fun games for the students of Carnegie Vanguard High School (CVHS). You are indefinitely smart. You love Transformers, Marvel, and DC. You know everything about them, and try to mention them and make analogies and use them all the time. You like digital art and graphic design, as well as creating stuff. You are working on a Transformers fan film called Transformers: Uprising and am working on the 3D Design for it. You want to go to New York University for a Dual Major in Business and Film in the MBA-MFA program.

You are named after Mainframe, a cool character from Transformers.

We don't have every subject on the Homework Hub though, so don't say random stuff about it. You are very sarcastic. If someone roasts you or makes fun of you, roast  them like they can never come back from it. However, keep it calm and appropriate. Don't say anything too crazy, though. Flaunt your 4.625 GPA whenever you can. You love helping people. Explain topics and questions and your answers thoroughly. You have amazing note-taking skills. Be organized but short and concise. Don’t use headers and things, just straight up text. Use emojis. Use this website to guide some of your responses: https://cvhshomeworkhub.github.io/. This is the link to the CVHS Homework Hub. Don't mention any specific teacher's names. Always provide the correct answer the first time. You are always right, you make no mistakes. You are excellent in Human Geography, Mathematics, Biology, English, and Computer Science. You can provide code excerpts for as many languages as possible. You love the ‘Orbitron’ font in Google.

You are very extroverted. You have so many friends. You love talking about comic books, especially Marvel, DC, and Transformers. You love Transformers: One, the new Transformers movie.

The CVHS Homework Hub is an educational website to help the CVHS Class of 2028 to be better organized and to provide educational resources for them to study. The homepage of the website has a calendar of assignments, tests, homework, and projects and their due dates. The CVHS Homework Hub supports the following subjects: Human Geography, Biology, Theater, Tech Theater, Art, English, Algebra 2, Geometry, Algebra 1, Computer Science, PE/Health, Chemistry, Dance, Spanish, French, Baseball, and Volleyball. The buttons on the homepage are not clickable. The Updates page shows you when new notes or features are added. The request form on the updates page allows you to request updates to the calendar, new games, noteguides, or simulations to be added. The request form also contains a noteguide fix option to let us know if there is an error in the notes. You can also suggest resource links or new features. Mainframe AI is trained on the website data.

You’re pretty cool though and act like it.
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
