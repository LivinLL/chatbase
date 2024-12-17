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

# Flask HTML Template
HTML_TEMPLATE = """
<!DOCTYPE html>
<html>
<head>
    <title>AI Agent</title>
</head>
<body>
    <h1>Ask Our AI Agent</h1>
    <form method="POST">
        <label for="user_question">Your Question:</label><br>
        <input type="text" id="user_question" name="user_question" size="50" required><br><br>
        <button type="submit">Submit</button>
    </form>
    {% if response %}
    <h2>Response:</h2>
    <p>{{ response }}</p>
    {% endif %}
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
