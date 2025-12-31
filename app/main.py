from flask import Flask, request, render_template, jsonify
from models.vector_store import VectorStore
from services.llm_service import LLMService
from services.storage_service import S3Storage
from config import Config
import os
from langchain.document_loaders import TextLoader,PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
import tempfile
import logging


app = Flask(__name__)
vector_store = VectorStore(Config.VECTOR_DB_PATH)
storage_service = S3Storage()
llm_service = LLMService(vector_store)


@app.route('/')
def index():
    return render_template('index.html')


#Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def process_document(file):
    """Process file based on its type and return text chunks."""
    temp_dir = tempfile.mkdtemp()
    temp_path = os.path.join(temp_dir,file.filename)

    try:
        ## save file temporarily
        file.save(temp_path)

        #process based on file type
        if file.filename.endswith('.pdf'):
            loader = PyPDFLoader(temp_path)
            documents = loader.load()
        elif file.filename.endswith('.txt'):
            loader = TextLoader(temp_path)
            documents = loader.load()
        else:
            raise ValueError("Unsupported file type")

        ##split text into chunks
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=200
        )      

        text_chunks = text_splitter.split_documents(documents)
        return text_chunks
    finally:
        #clean up temporary file
        if os.path.exists(temp_path):
            os.remove(temp_path)
        os.rmdir(temp_dir)

@app.route('/upload', methods=['POST'])
def upload_file():
    try:
        logger.debug("Upload endpoint called")

        if 'file' not in request.files:
            logger.warning("No file part in the request")
            return jsonify({"error":" No file part in the request"}),400

        file = request.files['file']
        if file.filename == '':
            logger.warning("No selected file")
            return jsonify({"error":" No selected file"}),400

        # check file extension
        if not file.filename.endswith(('.pdf','.txt')):
            logger.warning("Unsupported file type")
            return jsonify({"error":" Unsupported file type"}),400
        
        #process document
        try:
            text_chunks = process_document(file)
            logger.debug(f"Processed {len(text_chunks)} text chunks from the document")
        except Exception as e:
            logger.error(f"Error processing document: {str(e)}")
            return jsonify({"error": f"Error processing document: {str(e)}"}),500

        #Upload original file to s3
        try:
            file.seek(0)  #reset file pointer
            storage_service.upload_file(file,file.filename)
            logger.debug("File uploaded to s3 successfully")
        except Exception as e:
            logger.error(f"Error uploading file to s3: {str(e)})")
            return jsonify({"error": f"Error uploading file to s3: {str(e)}"}),500

        #Add documents to vector store
        try:
            vector_store.add_documents(text_chunks)
            logger.debug("Documents added to vector store successfully")
        except Exception as e:
            logger.error(f"Error adding documents to vector store: {str(e)}")
            return jsonify({"error": f"Error adding documents to vector store: {str(e)}"}),500

    except Exception as e:
        logger.error(f"Unexpected error: {str(e)}")
        return jsonify({"error": f"Unexpected error: {str(e)}"}),500
    

@app.route('/query', methods=['POST'])
def query():
    data = request.json
    if 'question' not in data:
        return jsonify({"error": "No question provided"}),400
    
    try:
        response = llm_service.get_response(data['question'])
        return jsonify({"response": response})
    except Exception as e:
        logger.error(f"Error processing query: {str(e)}")
        return jsonify({"error": f"Error processing query: {str(e)}"}),500


if __name__ == "__main__":
    app.run(host='0.0.0.0',port=5000,debug=True)