import os
import openai
import pinecone
from langchain.llms import OpenAI
from langchain.chains.question_answering import load_qa_chain
import sys
from flask import Flask, render_template, request
from werkzeug.utils import secure_filename
from langchain.document_loaders import DirectoryLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.vectorstores import Pinecone
from langchain.llms import OpenAI
from langchain.chat_models import ChatOpenAI
from langchain.chains.question_answering import load_qa_chain
from flask_limiter import Limiter
from flask_limiter import Limiter
from flask_limiter.util import get_remote_address
import os
import time

app = Flask(__name__)
limiter = Limiter(key_func=get_remote_address)
limiter.init_app(app)
app.config['UPLOAD_FOLDER'] = 'UPLOAD_FOLDER'
app.config['ALLOWED_EXTENSIONS'] = {'pdf'}

api_key = os.environ.get("OPENAI_API_KEY")
index_name = "booktest"
embeddings = OpenAIEmbeddings(openai_api_key=api_key)
pinecone.init(api_key=os.environ.get("Pine_api"), environment=os.environ.get("Pine_env"))
model_name = "gpt-3.5-turbo"

class OpenAILabAssistant:
    def __init__(self, api_key, index_name, chain_type="stuff"):
        self.api_key = api_key
        self.index_name = index_name
        self.chain_type = chain_type
        self.pinecone_index = None
        self.chain = None

    def setup(self):
        self.pinecone_index = Pinecone.from_existing_index(self.index_name, embeddings)
        self.embeddings = OpenAIEmbeddings(openai_api_key=self.api_key)
        self.llm = ChatOpenAI(openai_api_key=api_key, temperature=0, model_name=model_name)
        self.chain = load_qa_chain(self.llm, chain_type=self.chain_type)

    def generate_text(self, prompt, max_tokens=50):
        prompt = "Answer only if the answer is in the document but give all the information you can: " + prompt
        response = self.llm.generate(prompt, max_tokens=max_tokens)
        generated_text = response.choices[0].text
        return generated_text

    def search_document(self, question):
        similar_docs = self.get_similar_docs(question)
        if similar_docs:
            answer = self.chain.run(input_documents=similar_docs, question=question)
            return answer
        else:
            return "Sorry, no relevant information found."

    def get_similar_docs(self, query, k=2, score=False):
        top_k = k if not score else k + 1
        if score:
            similar_docs = self.pinecone_index.similarity_search_with_score(query, k=top_k)
        else:
            similar_docs = self.pinecone_index.similarity_search(query, k=top_k)
        return similar_docs

assistant = OpenAILabAssistant(api_key=api_key, index_name=index_name)
assistant.setup()

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in app.config['ALLOWED_EXTENSIONS']

def load_docs(directory):
    loader = DirectoryLoader(directory)
    documents = loader.load()
    return documents

def split_docs(documents, chunk_size=1000, chunk_overlap=0):
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
    docs = text_splitter.split_documents(documents)
    return docs

@limiter.limit("200/day;50/hour")
@app.route("/", methods=['GET', 'POST'])
def home():
    question_answer = ''
    file_message = ''
    if request.method == 'POST':
        if 'question' in request.form and request.form['question'].strip():
            question = request.form['question']
            answer = assistant.search_document(question)
            question_answer = f'{question}:{answer}'


        # File processing
        if 'file' in request.files:
            file = request.files['file']
            # Check if the file is not None and the filename is not an empty string
            if file and file.filename != '':
                if allowed_file(file.filename):
                    filename = secure_filename(file.filename)
                    file.save(os.path.join(app.config['UPLOAD_FOLDER'], filename))

                    documents = load_docs(app.config['UPLOAD_FOLDER'])
                    docs = split_docs(documents)

                    # Add documents to Pinecone
                    index = Pinecone.from_documents(docs, embeddings, index_name=index_name)
                    os.remove(os.path.join(app.config['UPLOAD_FOLDER'], filename))

                    file_message = "File processed successfully"
                else:
                    file_message = "File must be a pdf"

    return render_template("Index.html", question_answer=question_answer, file_message=file_message)


if __name__ == "__main__":
    app.run(debug=True)








