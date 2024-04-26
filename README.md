# AI Document Navigator: Navigate Your Documents with OPENAI GPT 3.5 and Llamaindex

Welcome to the AI Document Navigator project! This application demonstrates the power of AI in simplifying document search and retrieval. Using OpenAI's GPT-3.5 and LlamaIndex, it allows users to find relevant documents through natural language queries.

The application workflow involves:
- Document Processing: PDFs are read, and text is extracted for indexing and analysis.
- Document Summary Index: LlamaIndex creates an index of documents along with their summaries for efficient retrieval.
- Query Processing: The user's natural language query is interpreted using the LLM.
- Relevant Documents Retrieval: The system retrieves the most relevant documents based on the query and their summaries.
- Justification Generation: The LLM explains why each retrieved document is relevant to the user's query.
- Results Presentation: The interface displays document names, summaries, and justifications in a clear and organized manner.

## Demo Features
- **Natural Language Processing**: This tool understands natural language queries, allowing users to search for documents as easily as asking a question.
- **Document Retrieval and Summarization**: It retrieves document names, provides summaries, and explains why these documents are relevant to the query, making it easier for users to decide which documents to delve into.
- **User-Friendly Interface**: Built with Streamlit, the interface is straightforward, featuring options that guide users through document retrieval without any hassle.

## Tools and Technologies Used
- LLM: OpenAI's GPT-3.5
- LLM Orchestration: Llama-Index
- UI Framework: Streamlit

## Project Structure
- `app.py`: The main application script for the Streamlit app.
- `Dockerfile`: Contains the Docker configuration for building and running the app.
- `requirements.txt`: Lists the Python packages required for this project.
- `.env`: File to include `OPENAI_API_KEY` for authentication.

## Setup and Usage

### Clone the Repository
```
git clone https://github.com/Cohorte-ai/ask_and_get_demo/
```

### Install Required Packages

```
pip install -r requirements.txt
```

### Run the Streamlit App
```
streamlit run app.py
```

### Docker Support
To build and run the application using Docker, follow these steps:

#### Build the Docker Image
```
docker build -t ask_and_get_demo .
```
#### Run the Docker Container
```
docker run -p 8501:8501 --env-file .env ask_and_get_demo
```
Note: Ensure you have the `.env` file containing `OPENAI_API_KEY` in your project directory before running the Docker container.

