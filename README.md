# Ollam-Play - A RAG Application

This application is a playful exploration of Ollama, LangChain, LLaMa, and other LLM tools. It enables users to interact with product information from a CSV file via a chat interface, utilizing LangChain for document retrieval and a custom LLaMa model for natural language processing.

## Prerequisites

Before running the application, ensure that you have the following installed:

- Python 3.7 or higher
- pip (Python package manager)
- **Ollama Server** running locally with the required models pulled from Ollama (LLaMa, or any other models you wish to use)

You can download and install Ollama from [Ollama's official website](https://ollama.com/). Once installed, ensure you have the models pulled and ready by running:

```bash
ollama pull llama
```

This application makes API calls to the Ollama server running on `http://localhost:11434/api/generate`.

## Installation

### Step 1: Clone the repository

First, clone the repository to your local machine:

```bash
git clone https://github.com/Tswalano/ollama-play.git
cd ollama-play
```

### Step 2: Set up a virtual environment (optional but recommended)

It's a good practice to use a virtual environment to isolate the project dependencies:

```bash
python3 -m venv venv
source venv/bin/activate  # On Windows, use `venv\Scripts\activate`
```

### Step 3: Install dependencies

Run the following command to install all the required Python packages listed in `requirements.txt`:

```bash
pip install -r requirements.txt
```

### Step 4: Ensure the Ollama API is running

Make sure that the Ollama server is running locally and that the required models are loaded. The application will make API calls to the LLaMa model (or whichever model you have pulled) at the following endpoint:

```
http://localhost:11434/api/generate
```

If you haven’t set up the Ollama API yet, you can follow the setup instructions from [Ollama's documentation](https://ollama.com/docs/).

### Step 5: Run the Flask application

Now you can run the Flask app:

```bash
python app.py
```

By default, the Flask app will run on `http://127.0.0.1:5000`.

### Step 6: Interacting with the application

Once the application is running:

- Open your browser and navigate to `http://127.0.0.1:5000/` to view the application interface.
- Send POST requests to the `/chat` endpoint with the query in JSON format, for example:

```json
{
  "query": "How much is the wireless mouse?"
}
```

The server will respond with the product details, such as the price and description.

### Step 7: Testing the application

You can also test the application by sending a POST request via `curl`, Postman, or any HTTP client:

```bash
curl -X POST http://127.0.0.1:5000/chat \
  -H "Content-Type: application/json" \
  -d '{"query": "How much is the wireless mouse?"}'
```

### Troubleshooting

- **Missing dependencies**: Ensure all dependencies are correctly installed by running `pip install -r requirements.txt`.
- **Ollama server not running**: Make sure the Ollama server is running locally and that the models are properly loaded and accessible via the endpoint `http://localhost:11434/api/generate`.
- **Invalid model response**: If the Ollama API is not returning the expected results, double-check the model loading process and ensure the correct model is being used.

### License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.