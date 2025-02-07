from flask import Flask, request, jsonify, Response
from langchain_huggingface import HuggingFaceEmbeddings, HuggingFaceEndpoint
from langchain.vectorstores import FAISS
from langchain.chains import RetrievalQA
import os
import json
import logging
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Initialize Flask app
app = Flask(__name__)

# Set up logging
logging.basicConfig(level=logging.INFO)

# Ensure FAISS index exists before loading
FAISS_INDEX_PATH = "faiss_index"
if not os.path.exists(FAISS_INDEX_PATH):
    logging.error(f"FAISS index directory '{FAISS_INDEX_PATH}' not found.")
    raise FileNotFoundError(f"FAISS index directory '{FAISS_INDEX_PATH}' not found. Ensure it is uploaded.")

# Load stored vector database (FAISS)
embedding_model = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
vectorstore = FAISS.load_local(FAISS_INDEX_PATH, embedding_model, allow_dangerous_deserialization=True)

# Use environment variable for API key
HUGGINGFACE_API_KEY = os.getenv("HUGGINGFACEHUB_API_TOKEN")
if not HUGGINGFACE_API_KEY:
    logging.error("Hugging Face API key is missing.")
    raise ValueError("Hugging Face API key is missing. Set it as an environment variable in the .env file.")

# Initialize Hugging Face Language Model (LLM) with explicit task
llm = HuggingFaceEndpoint(
    repo_id="tiiuae/falcon-7b-instruct",
    task="text-generation",  # Explicitly defining the task
    huggingfacehub_api_token=HUGGINGFACE_API_KEY
)

# Create a Retrieval-based QA System
qa = RetrievalQA.from_chain_type(llm=llm, retriever=vectorstore.as_retriever())

@app.route("/chat", methods=["POST"])
def chat():
    try:
        data = request.get_json()
        user_query = data.get("query", "").strip()

        if not user_query:
            return jsonify({"error": "No query provided"}), 400

        logging.info(f"Received query: {user_query}")

        # Retrieve the most relevant response from stored course data
        response = qa.invoke(user_query)  # ✅ Fixed deprecation issue

        # Format response for better readability
        formatted_response = json.dumps({"response": response}, indent=4)
        return Response(formatted_response, mimetype="application/json")

    except Exception as e:
        logging.error(f"Error processing request: {str(e)}")
        return jsonify({"error": "Something went wrong. Please try again later."}), 500

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5003, debug=False)
