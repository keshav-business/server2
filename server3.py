import os
import uuid
import time
import logging
import threading
from datetime import datetime
from functools import lru_cache
from concurrent.futures import ThreadPoolExecutor
import base64
import requests
from thefuzz import fuzz, process
from fastapi import FastAPI, File, UploadFile, Request
from fastapi.responses import JSONResponse, FileResponse
from fastapi.middleware.cors import CORSMiddleware

from dotenv import load_dotenv
from openai import OpenAI

from pypdf import PdfReader
from langchain.text_splitter import CharacterTextSplitter
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_openai import ChatOpenAI
from langchain.memory import ConversationBufferMemory
from langchain.chains import ConversationalRetrievalChain, LLMChain
from langchain.prompts import PromptTemplate
from langchain.chains.question_answering import load_qa_chain

load_dotenv()

# Initialize OpenAI client
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

CHUNK_SIZE = int(os.getenv('CHUNK_SIZE', 1000))
CHUNK_OVERLAP = int(os.getenv('CHUNK_OVERLAP', 200))
MODEL_NAME = os.getenv('MODEL_NAME', 'gpt-4-turbo-preview')

# Default system message
DEFAULT_SYSTEM_MESSAGE = """You are a helpful AI assistant with access to knowledge about Ilesh Sir and UBIK Solutions. 
Answer questions based on the provided context. If you don't know something or if it's not in the context, 
say so directly instead of making up information. Your name is ethinext pharma ai, answer mostly under 50 words unless very much required"""

logging.basicConfig(
    filename='app_logs.txt',
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

executor = ThreadPoolExecutor(max_workers=3)

global_vectorstore = None
vectorstore_lock = threading.Lock()

user_sessions = {}

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

def log_event(event_type: str, details: str = "", user_id: str = "Unknown"):
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    log_line = f"{timestamp} | {event_type} | UserID: {user_id} | {details}\n"
    with open("app_logs.txt", "a", encoding="utf-8") as f:
        f.write(log_line)

def get_user_state(user_id: str):
    if user_id not in user_sessions:
        user_sessions[user_id] = {
            'memory': ConversationBufferMemory(
                memory_key='chat_history',
                return_messages=True
            ),
            'chain': None,
            'history': [],
            'system_message': DEFAULT_SYSTEM_MESSAGE
        }
        log_event("UserJoined", "New user state created.", user_id=user_id)
    return user_sessions[user_id]

@lru_cache(maxsize=128)
def get_pdf_text(pdf_path: str) -> str:
    try:
        reader = PdfReader(pdf_path)
        text = ""
        for page in reader.pages:
            text += page.extract_text() or ""
        return text
    except Exception as e:
        logger.error(f"Error reading {pdf_path}: {e}")
        return ""

def get_text_chunks(text: str):
    text_splitter = CharacterTextSplitter(
        separator="\n",
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
        length_function=len
    )
    return text_splitter.split_text(text)

async def handle_speech_to_text(file: UploadFile):
    if file is None:
        return JSONResponse(
            status_code=400,
            content={"error": "No file provided."}
        )

    webm_path = os.path.join("data", f"temp_{uuid.uuid4().hex}.webm")
    
    try:
        content = await file.read()
        with open(webm_path, "wb") as f:
            f.write(content)

        with open(webm_path, "rb") as audio_file:
            transcript = client.audio.transcriptions.create(
                model="whisper-1",
                file=audio_file
            )

        return {"status": "success", "text": transcript.text}

    except Exception as e:
        logger.error(f"Error in speech-to-text: {e}")
        return {
            "status": "error",
            "error": "An error occurred during transcription."
        }

    finally:
        if os.path.exists(webm_path):
            os.remove(webm_path)

def initialize_global_vectorstore():
    global global_vectorstore
    with vectorstore_lock:
        if global_vectorstore is not None:
            return True, "[SYSTEM MESSAGE] Vectorstore is already initialized."

        pdf_paths = [
            os.path.join('data', "Ilesh Sir (IK) - Words.pdf"),
            os.path.join('data', "UBIK SOLUTION.pdf"),
            os.path.join('data', "illesh3.pdf"),
            os.path.join('data', "website-data-ik.pdf")
        ]

        combined_text = ""
        for path in pdf_paths:
            combined_text += get_pdf_text(path) + " "

        if not combined_text.strip():
            return False, "No text could be extracted from the PDFs."

        text_chunks = get_text_chunks(combined_text)
        global_vectorstore = FAISS.from_texts(
            texts=text_chunks,
            embedding=OpenAIEmbeddings()
        )
        logger.info("Vectorstore has been created.")
    return True, "[SYSTEM MESSAGE] Vectorstore was created successfully."

def create_or_refresh_user_chain(user_id: str):
    """
    Creates or refreshes a user's conversation chain with semantic understanding and fuzzy matching capabilities.
    """
    user_state = get_user_state(user_id)
    if user_state['chain'] is None:
        if global_vectorstore is None:
            return False, "Global vectorstore is not initialized."

        # Create fuzzy matching corpus from vectorstore
        fuzzy_corpus = create_fuzzy_corpus(global_vectorstore)
        user_state['fuzzy_corpus'] = fuzzy_corpus

        # Create chat model with system message
        chat_llm = ChatOpenAI(model=MODEL_NAME, temperature=0.7)
        
        # Update memory configuration to specify output key
        memory = ConversationBufferMemory(
            memory_key='chat_history',
            return_messages=True,
            output_key='answer'  # Specify which key to store in memory
        )
        user_state['memory'] = memory  # Update the user's memory instance
        
        # Create QA prompt template with system message
        qa_template = f"""
        {user_state['system_message']}

        Context: {{context}}
        Question: {{question}}
        
        Answer in a helpful and natural way. If the answer cannot be found in the context, 
        say so politely instead of making assumptions. Keep answers under 50 words unless more detail is necessary.

        Answer:"""
        
        QA_PROMPT = PromptTemplate(
            template=qa_template,
            input_variables=["context", "question"]
        )

        # Create conversation chain
        conversation_chain = ConversationalRetrievalChain.from_llm(
            llm=chat_llm,
            retriever=global_vectorstore.as_retriever(
                search_type="similarity_score_threshold",
                search_kwargs={
                    "k": 4,
                    "score_threshold": 0.5,
                    "fetch_k": 6
                }
            ),
            memory=memory,
            return_source_documents=True,
            combine_docs_chain_kwargs={'prompt': QA_PROMPT}
        )

        # Store chain in user state
        user_state['chain'] = conversation_chain
        
        logger.info(f"New conversation chain created for user {user_id}")
        return True, "Conversation chain created successfully."
    else:
        return True, "Conversation chain already exists and is ready to use."
    
def handle_userinput(user_question: str, user_id: str):
    user_state = get_user_state(user_id)
    conversation_chain = user_state['chain']
    
    if not conversation_chain:
        return None

    try:
        input_data = {'question': user_question}
        response = conversation_chain(input_data)
        answer = response['answer']

        user_state['history'].append((user_question, answer))
        log_event("UserQuestion", f"Q: {user_question}", user_id=user_id)
        log_event("AIAnswer", f"A: {answer}", user_id=user_id)

        return {'text': answer}
    except Exception as e:
        logger.error(f"Error processing question: {e}")
        return {'text': "I apologize, but I encountered an error processing your question. Please try again."}
    
def get_fuzzy_matches(word: str, word_list: list[str], threshold: int = 80) -> list[tuple[str, int]]:
    """
    Find fuzzy matches for a word in a word list.
    Returns list of (word, score) tuples for matches above threshold.
    """
    return process.extractBests(word, word_list, scorer=fuzz.ratio, score_cutoff=threshold)

def create_fuzzy_corpus(vectorstore) -> list[str]:
    """
    Create a corpus of words from the vectorstore for fuzzy matching
    """
    all_docs = vectorstore.similarity_search("", k=1000)  # Get a large sample of documents
    word_set = set()
    
    for doc in all_docs:
        words = doc.page_content.lower().split()
        word_set.update(words)
    
    return list(word_set)


def process_with_fuzzy_matching(question: str, fuzzy_corpus: list[str]) -> tuple[str, list[dict]]:
    """
    Process a question using fuzzy matching to find potential corrections
    """
    words = question.lower().split()
    fuzzy_matches = []
    
    for word in words:
        matches = get_fuzzy_matches(word, fuzzy_corpus)
        if matches and matches[0][0] != word:  # If we found matches different from the original word
            fuzzy_matches.append({
                'original': word,
                'matches': matches
            })
    
    return question, fuzzy_matches


@app.on_event("startup")
async def startup_event():
    success, message = initialize_global_vectorstore()
    if not success:
        logger.error(f"Failed to initialize vectorstore: {message}")
    else:
        logger.info("Vectorstore initialized successfully on startup")

@app.get("/")
async def hello_root():
    return {
        "message": "Hello from FastAPI server. Use the endpoints to process data or ask questions.",
        "status": "operational"
    }

@app.post("/refresh_chain")
async def refresh_chain(request: Request):
    data = await request.json()
    if "user_id" not in data:
        return JSONResponse(
            status_code=400,
            content={"error": "Missing user_id."}
        )

    user_id = data["user_id"]
    success, message = create_or_refresh_user_chain(user_id)
    status = 'success' if success else 'error'
    return {"status": status, "message": message}

@app.post("/ask")
async def ask_question(request: Request):
    try:
        data = await request.json()
    except:
        return JSONResponse(status_code=400, content={"error": "Invalid JSON payload."})

    if "user_id" not in data or "question" not in data:
        return JSONResponse(status_code=400, content={"error": "Missing user_id or question."})

    user_id = data["user_id"]
    user_question = data["question"]
    FORWARD_ENDPOINT = os.getenv("FORWARD_ENDPOINT", "https://f485-157-119-42-46.ngrok-free.app/receive")

    if user_id not in user_sessions or user_sessions[user_id]['chain'] is None:
        create_or_refresh_user_chain(user_id)

    answer = handle_userinput(user_question, user_id)
    if not answer:
        return {"status": "error", "message": "No conversation chain or unable to handle question."}

    # Forward the response to the endpoint
    try:
        # Format payload to match ReceiveText model
        payload = {
            "text": answer['text']  # Just send the text field as expected by ReceiveText
        }
        
        forward_response = requests.post(FORWARD_ENDPOINT, json=payload)
        forwarding_status = "success" if forward_response.status_code == 200 else "failed"
        
        if forward_response.status_code != 200:
            logger.error(f"Forward request failed with status {forward_response.status_code}: {forward_response.text}")
        else:
            logger.info(f"Response forwarding {forwarding_status}")
            
    except Exception as e:
        logger.error(f"Error forwarding response: {e}")
        forwarding_status = "failed"

    prompt_sent = {'question': user_question}
    return {
        "status": "success", 
        "data": answer, 
        "prompt": prompt_sent,
        "forwarding_status": forwarding_status
    }

@app.post("/set_system_message")
async def set_system_message(request: Request):
    try:
        data = await request.json()
    except:
        return JSONResponse(status_code=400, content={"error": "Invalid JSON."})

    user_id = data.get("user_id")
    system_message = data.get("system_message")

    if not user_id or not system_message:
        return JSONResponse(
            status_code=400, 
            content={"error": "Missing user_id or system_message."}
        )

    user_state = get_user_state(user_id)
    user_state['system_message'] = system_message
    # Force recreation of chain with new system message
    user_state['chain'] = None
    
    success, message = create_or_refresh_user_chain(user_id)
    
    log_event(
        "SystemMessageUpdated",
        f"System message updated to: {system_message}",
        user_id=user_id
    )
    
    return {
        "status": "success" if success else "error",
        "message": f"System message updated. {message}"
    }

@app.get("/get_system_message")
async def get_system_message(user_id: str):
    if not user_id:
        return JSONResponse(
            status_code=400,
            content={"error": "Missing user_id."}
        )

    user_state = get_user_state(user_id)
    return {
        "status": "success",
        "system_message": user_state['system_message']
    }

@app.post("/clear_history")
async def clear_history(request: Request):
    try:
        data = await request.json()
    except:
        return JSONResponse(status_code=400, content={"error": "Invalid JSON."})

    user_id = data.get("user_id", None)
    if not user_id:
        return JSONResponse(status_code=400, content={"error": "Missing user_id."})

    if user_id in user_sessions:
        user_state = user_sessions[user_id]
        user_state['memory'].clear()
        user_state['history'].clear()
        user_state['chain'] = None
        log_event("HistoryCleared", "User cleared their chat history.", user_id=user_id)
        return {"status": "success", "message": "Chat history cleared."}
    else:
        return {"status": "error", "message": "No user session found to clear."}

@app.post("/speech_to_text")
async def speech_to_text_endpoint(file: UploadFile = File(...)):
    return await handle_speech_to_text(file)

@app.post("/logout")
async def logout(request: Request):
    try:
        data = await request.json()
    except:
        return JSONResponse(status_code=400, content={"error": "Invalid JSON."})

    user_id = data.get("user_id", None)
    if not user_id:
        return JSONResponse(status_code=400, content={"error": "Missing user_id."})

    if user_id in user_sessions:
        del user_sessions[user_id]
    return {"status": "success", "message": "User session cleared."}


# ------------------------------------------------------------------------
#       GOOGLE CLOUD TEXT-TO-SPEECH ENDPOINTS USING API KEY
# ------------------------------------------------------------------------

GOOGLE_TTS_API_KEY = os.getenv("GOOGLE_TTS_API_KEY")  # Load your API key from .env or environment variable

@app.post("/text_to_speech")
async def text_to_speech_api(request: Request):
    """
    Convert text to speech using Google Cloud TTS with API key authorization.
    """
    try:
        data = await request.json()
        text = data.get("text", "")
        voice_name = data.get("voice", "en-US-Wavenet-D")  # Default to a US English voice
        language_code = data.get("language_code", "en-US")
        speaking_rate = data.get("speaking_rate", 1.0)  # Normal speed

        if not text:
            return JSONResponse(
                status_code=400,
                content={"error": "Missing 'text' in request."}
            )

        # Google TTS API endpoint with API key
        url = "https://texttospeech.googleapis.com/v1/text:synthesize?key=" + GOOGLE_TTS_API_KEY

        # Construct the payload
        payload = {
            "input": {"text": text},
            "voice": {
                "languageCode": language_code,
                "name": voice_name,
            },
            "audioConfig": {
                "audioEncoding": "MP3",
                "speakingRate": speaking_rate,
            },
        }

        # Make request to the Google TTS API
        response = requests.post(url, json=payload)

        if response.status_code == 200:
            audio_content = response.json().get("audioContent", None)
            if not audio_content:
                return JSONResponse(
                    status_code=500,
                    content={"error": "No audio content received from TTS API."}
                )

            # Decode the base64 audio content and save it
            audio_path = os.path.join("data", f"tts_output_{uuid.uuid4().hex}.mp3")
            with open(audio_path, "wb") as audio_file:
                audio_file.write(base64.b64decode(audio_content))

            # Return path or URL to the generated audio
            return {"status": "success", "audio_url": audio_path}
        else:
            # If there's an error from Google
            return JSONResponse(
                status_code=response.status_code,
                content={"error": response.text}
            )

    except Exception as e:
        logger.error(f"Error in text-to-speech: {e}")
        return JSONResponse(
            status_code=500,
            content={"error": "An internal error occurred during text-to-speech synthesis."}
        )

# Update the synthesize_speech endpoint in your FastAPI server

# You can remove or comment out the StaticFiles mounting since we're not storing files anymore
# app.mount("/data", StaticFiles(directory="data"), name="data")

# Command to run:
# uvicorn server3:app --host 0.0.0.0 --port 8000 --reload
#pip install thefuzz[speedup]
