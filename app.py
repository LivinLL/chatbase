import os
from flask import Flask, request, render_template_string
from pinecone import Pinecone
import openai

# Load environment variables
PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
PINECONE_ENVIRONMENT = os.getenv("PINECONE_ENVIRONMENT")
PINECONE_INDEX_NAME = os.getenv("PINECONE_INDEX_NAME")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

# Initialize Flask app
app = Flask(__name__)

# Connect to Pinecone
pc = Pinecone(api_key=PINECONE_API_KEY)
index = pc.Index(PINECONE_INDEX_NAME)

# OpenAI setup
openai.api_key = OPENAI_API_KEY

# Flask HTML Template with updated UI
HTML_TEMPLATE = """
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Ask d-A-v-I-d about LLU Stuff</title>
    <style>
        body {
            font-family: Arial, sans-serif;
            margin: 0;
            padding: 0;
            background-color: #f9f9f9;
            display: flex;
            justify-content: center;
            align-items: center;
            height: 100vh;
            color: #333;
        }

        .container {
            width: 100%;
            max-width: 700px;
            margin: 0 auto;
            background: white;
            padding: 20px;
            border-radius: 8px;
            box-shadow: 0 4px 8px rgba(0, 0, 0, 0.1);
            position: relative;
        }

        h1 {
            text-align: center;
            font-size: 24px;
            color: #2e6da4;
        }

        form {
            display: flex;
            flex-direction: column;
            align-items: center;
            padding: 0 15px; /* Padding on mobile */
        }

        label, input {
            margin-bottom: 10px;
        }

        input[type="text"] {
            padding: 10px;
            width: 100%;
            max-width: 600px;
            border-radius: 5px;
            border: 1px solid #ccc;
            font-size: 16px;
        }

        button {
            padding: 12px 20px;
            background-color: #2e6da4;
            color: white;
            border: none;
            border-radius: 5px;
            cursor: pointer;
            font-size: 16px;
            transition: background-color 0.3s ease;
        }

        button:hover {
            background-color: #22597e;
        }

        /* Spinner styles */
        #loading-spinner {
            display: none;
            margin: 20px auto;
            width: 50px;
            height: 50px;
            border: 5px solid #f3f3f3;
            border-top: 5px solid #3498db;
            border-radius: 50%;
            animation: spin 1s linear infinite;
        }

        @keyframes spin {
            0% {
                transform: rotate(0deg);
            }
            100% {
                transform: rotate(360deg);
            }
        }

        /* Center spinner */
        #loading-container {
            text-align: center;
        }

        #loading-overlay {
            display: none;
            position: fixed;
            top: 0;
            left: 0;
            width: 100%;
            height: 100%;
            background-color: rgba(0, 0, 0, 0.5); /* Dark overlay */
            z-index: 500; /* Ensure it is on top */
        }

        /* Responsive design */
        @media (max-width: 767px) {
            .container {
                padding: 15px;
            }

            h1 {
                font-size: 20px;
            }

            input[type="text"] {
                font-size: 14px;
            }

            button {
                font-size: 14px;
                padding: 10px;
            }
        }
    </style>
</head>
<body>
    <div class="container">
        <h1>Ask d-A-v-I-d about LLU Stuff</h1>
        <form method="POST" id="questionForm">
            <label for="user_question">Your Question:</label><br>
            <input type="text" id="user_question" name="user_question" size="50" required><br><br>
            <button type="submit" id="submitButton">Ask d-A-v-I-d</button>
        </form>

        <!-- Loading Spinner and Overlay -->
        <div id="loading-container">
            <div id="loading-spinner"></div>
            <p id="loading-text" style="display: none;">d-A-v-I-d is thinking...</p>
        </div>
        <div id="loading-overlay"></div>

        {% if response %}
        <h2>d-A-v-I-d says:</h2>
        <p>{{ response }}</p>
        {% endif %}
    </div>

    <script>
        const form = document.getElementById('questionForm');
        const submitButton = document.getElementById('submitButton');
        const loadingSpinner = document.getElementById('loading-spinner');
        const loadingOverlay = document.getElementById('loading-overlay');
        const loadingText = document.getElementById('loading-text');

        form.onsubmit = function(event) {
            event.preventDefault();  // Prevent default form submission

            // Show the loading spinner and overlay
            loadingSpinner.style.display = 'block';
            loadingText.style.display = 'block';
            loadingOverlay.style.display = 'block';
            submitButton.disabled = true;  // Disable the button

            // Submit the form using AJAX or let Flask handle the POST
            form.submit();
        };
    </script>
</body>
</html>
"""

# Query Pinecone and generate GPT-4 response
def query_pinecone_and_generate_response(user_query):
    # Load the system prompt from a file
    with open("prompt.txt", "r") as file:
        system_prompt = file.read()

    # Generate embedding for the user query
    response = openai.embeddings.create(input=user_query, model="text-embedding-ada-002")
    query_embedding = response.data[0].embedding

    # Query Pinecone for the top 3 most similar chunks
    search_results = index.query(vector=query_embedding, top_k=3, include_metadata=True)
    relevant_chunks = [match.metadata['text'] for match in search_results['matches']]

    # Combine chunks into context for GPT-4
    context = "\n\n".join(relevant_chunks)
    full_prompt = f"{system_prompt}\n\n### Context:\n{context}\n\nQuestion: {user_query}\nAnswer:"

    # Generate GPT-4 response
    gpt_response = openai.chat.completions.create(
        model="gpt-4",
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": full_prompt}
        ]
    )

    # Return the GPT-4 response
    return gpt_response.choices[0].message.content


# Flask route
@app.route("/", methods=["GET", "POST"])
def home():
    if request.method == "POST":
        user_question = request.form["user_question"]
        response = query_pinecone_and_generate_response(user_question)
        return render_template_string(HTML_TEMPLATE, response=response)
    return render_template_string(HTML_TEMPLATE)

# Production-ready app run
if __name__ == "__main__":
    # Render will pass the PORT environment variable; use it or default to 5000 for local
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port)
