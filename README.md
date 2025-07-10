# Langchain Flask Bot

A Flask-based chatbot application that leverages LangChain, Hugging Face models, and FAISS vector search to answer questions based on technical course data scraped from the web.

## Features
- **Web Scraping & Indexing:** Scrapes course data from a technical website and indexes it using FAISS for fast retrieval.
- **Semantic Search:** Uses embeddings to find the most relevant course information for user queries.
- **Conversational QA:** Integrates with Hugging Face language models to generate natural language answers.
- **REST API:** Exposes a `/chat` endpoint for programmatic access.

## Requirements
- Python 3.8+
- Hugging Face account and API token (for hosted inference)
- Internet connection (for scraping and model inference)

## Setup

1. **Clone the repository:**
   ```bash
   git clone <repo-url>
   cd Langchain_Flask_Bot
   ```

2. **Create and activate a virtual environment:**
   ```bash
   python3 -m venv env
   source env/bin/activate
   ```

3. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

4. **Set up environment variables:**
   - Create a `.env` file in the project root with:
     ```env
     HUGGINGFACEHUB_API_TOKEN=your_huggingface_api_token
     ```
   - (Optional) Set a custom user agent:
     ```env
     USER_AGENT=YourCustomUserAgentString
     ```

5. **Generate the FAISS index:**
   ```bash
   python generate_faiss.py
   ```
   - This will scrape course data and build the vector index.

6. **Start the Flask app:**
   ```bash
   python app.py
   ```
   - The app will be available at `http://127.0.0.1:5003`.

## Usage

### Chat Endpoint
- **URL:** `POST /chat`
- **Request Body:**
  ```json
  { "query": "What technical courses are available?" }
  ```
- **Response:**
  ```json
  { "response": "...answer from the bot..." }
  ```

You can test with `curl`:
```bash
curl -X POST http://127.0.0.1:5003/chat \
     -H "Content-Type: application/json" \
     -d '{"query": "What technical courses are available?"}'
```

## Troubleshooting
- **FAISS index not found:** Run `python generate_faiss.py` before starting the app.
- **Hugging Face API errors:** Ensure your API token is valid and the model you use is available for hosted inference.
- **Deprecation warnings:** Some imports may be deprecated; see the warning messages for upgrade instructions.
- **Connection issues:** Check your internet connection and that the target website is accessible.

## Customization
- To change the source website or model, edit `generate_faiss.py` and `app.py` accordingly.

## License
MIT 
