## Efficient Study

### Goal

### Data Preparation and Setup

### Application and code

Below are all libraries in the requirements.txt file along with their importance:

1. `streamlit`: Used to run our Streamlit application
2. `langchain-community`: Used to help preprocess documents, create embeddings, and generate answers to user questions
3. `openai`: Used to call OpenAI API
4. `chromadb`: Used to create our vector DB to store the embeddings?
5. `tiktoken`: Used to manage token limits when working with OpenAI API
6. `pymupdf`:Helps in preprocessing the pdf document the user uploads
7. `python-dotenv` : Used to help load secret environment variable, like OpenAI API key
8. `faiss-cpu`: Used to create our vector DB to store the embeddings
9.  `psycopg2-binary`: Library to connect to our PostgreSQL DB
10. `pymongo`: Library to connect to our MongoDB
11.  `gridfs`: Used to store files in PostgreSQL DB

Below are some steps to run our application:

1. `git clone https://github.com/VineethNareddy/EfficientStudy.git`: Use this command to clone the repo in your IDE
2. `pip install -r requirements.txt`: To install all libraries used in the project. Should take ~5 min if you never installed the libraries in the requirements.txt file before
3. PostgreSQL installation: Look below
4. `streamlit run main.py`: To launch our application. Should take a short time before a user can finally interact with our application.

Note: It's best if you run this app with the latest version of Python installed. We're not sure of this, but if you don't have the latest version, any version >= 3.12 should work.

### Code Documentation and References


1. `https://github.com/dataprofessor/langchain-ask-the-doc`: This repo helped our team understand the architecture behind what we are trying to build and serves as a foundation for the backend.
