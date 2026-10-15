# Medical AI Assistant
### Zidio Development Internship Project

## 1. Project Overview

The Medical AI Assistant is a web-based chatbot that provides general educational health information. It helps users explore common medical conditions, symptoms, causes, preventive measures, diagnostic tests, and general management.

The project combines natural language processing, machine learning, a structured medical knowledge base, contextual conversation handling, and an optional generative AI fallback.

**Important:** This application is an educational prototype. It is not a substitute for professional medical advice, diagnosis, or treatment.

## 2. Key Features

- **Intent Recognition:** Classifies user questions into seven medical information categories.
- **Medical Knowledge Retrieval:** Retrieves relevant information from a structured CSV knowledge base.
- **Contextual Memory:** Uses conversation context to understand follow-up questions.
- **Medical Scope Handling:** Handles unrelated questions without presenting them as medical answers.
- **Source Attribution:** Includes source names recorded in the knowledge base.
- **Conversation Analytics:** Displays intent distribution, confidence, sentiment, response times, and recent activity.
- **Web Interface:** Provides an AI Assistant, Overview, and Conversation Insights page.
- **Generative AI Integration:** Supports a Gemini-compatible API integration when the local response path requires it.

## 3. Technology Stack

- Python
- Flask
- Pandas and NumPy
- Scikit-learn
- TF-IDF and Logistic Regression
- HTML, CSS, and JavaScript
- CSV-based medical knowledge base
- OpenAI Python client with the configured Gemini-compatible endpoint

## 4. Main Project Components

- `app.py` — Flask web server and API routes
- `app/chatbot_pipeline.py` — Message processing and contextual conversation logic
- `app/intent_detection.py` — Intent classification
- `app/faq_response.py` — Medical knowledge retrieval
- `app/response_generator.py` — Response generation and medical safety instructions
- `app/gpt_client.py` — Configured generative AI client
- `data/medical/medical_knowledge.csv` — Structured medical knowledge
- `templates/` — HTML dashboard pages
- `static/css/style.css` — Dashboard styling
- `static/js/app.js` — Chat interface and analytics behavior
- `requirements.txt` — Python dependencies

## 5. Installation

### Prerequisites

- Python 3.12 or a compatible supported Python version
- Git (optional, for version control)
- A Gemini API key if generative AI fallback is required

### Setup

Clone the repository or open the project folder.

Create a virtual environment:

```bash
python -m venv .venv
```

Activate it on Windows PowerShell:

```powershell
.\.venv\Scripts\Activate.ps1
```

Install dependencies:

```powershell
python -m pip install -r requirements.txt
```

Configure the API key in a local `.env` file if required by the project. Never publish API keys or secrets.

## 6. Run the Application

From the project root, with the virtual environment activated, run:

```powershell
python app.py
```

Open the following pages in a browser:

- Assistant: `http://127.0.0.1:5000/assistant`
- Overview: `http://127.0.0.1:5000/overview`
- Conversation Insights: `http://127.0.0.1:5000/insights`
- Health endpoint: `http://127.0.0.1:5000/api/health`

Stop the local server with `Ctrl+C`.

## 7. API Endpoints

### GET `/api/health`

Checks whether the backend is running.

### POST `/api/chat`

Accepts a JSON request containing a message and optional conversation history.

Example request:

```json
{
  "message": "What are common symptoms of diabetes?",
  "conversation_history": ""
}
```

The response includes the answer, detected intent, confidence values, sentiment, topic, response time, and API status.

## 8. Testing

Test the application with questions such as:

- What is diabetes?
- What are common symptoms of diabetes?
- How can I prevent diabetes?
- What are its symptoms?
- What is quantum physics?

Verify that medical questions receive relevant responses, follow-up questions use context, and unrelated questions are handled appropriately.

## 9. Safety and Limitations

- Responses are for general educational purposes only.
- The assistant does not provide a personal diagnosis or prescribe medication.
- Users should consult qualified healthcare professionals for individual medical advice.
- Emergency symptoms require urgent professional assessment.
- Knowledge-base coverage is limited to the topics and content available in the dataset.
- Model confidence is a classifier score, not a guarantee of medical accuracy.
- Generative AI availability and response quality depend on the configured service.

## 10. Future Improvements

- Expand and medically review the knowledge base.
- Add automated unit and API tests.
- Improve analytics consistency and session reporting.
- Add stronger error handling and request logging.
- Deploy with production-grade hosting and configuration.
- Add user feedback and evaluation of answer quality.

## 11. Disclaimer

This project is developed for educational and internship purposes. It is not a certified medical device and must not be used as the sole basis for healthcare decisions.
