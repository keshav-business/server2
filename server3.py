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
from langchain_community.embeddings import HuggingFaceEmbeddings
from fastapi import FastAPI, File, UploadFile, Request
from fastapi.responses import JSONResponse, FileResponse
from fastapi.middleware.cors import CORSMiddleware
import speech_recognition as sr
from dotenv import load_dotenv
from openai import OpenAI

from pypdf import PdfReader
from langchain.text_splitter import CharacterTextSplitter
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_openai import ChatOpenAI
from langchain.memory import ConversationBufferMemory
from langchain.chains import ConversationalRetrievalChain
from langchain.prompts import PromptTemplate
from langchain.chains.question_answering import load_qa_chain

load_dotenv()

# Initialize OpenAI client
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

CHUNK_SIZE = int(os.getenv('CHUNK_SIZE', 1000))
CHUNK_OVERLAP = int(os.getenv('CHUNK_OVERLAP', 200))
MODEL_NAME = os.getenv('MODEL_NAME', 'gpt-3.5-turbo')
EMBEDDING_MODEL = "all-MiniLM-L6-v2"

# Default system message
DEFAULT_SYSTEM_MESSAGE = """strictly answer in english and dont go out of the context that is provided to u but please cosider that there will be major speeech recognition errors so please work accordingly, You are a helpful AI assistant with access to UBIK Solutions. 
Answer questions based on the provided context. If you don't know something or if it's not in the context, dont go out of context, stick to what user said and form answer on that with help of context
say so directly instead of making up information. Your name is ubik ai, answer mostly under 50 words unless very much required, send back data in a clean format"""

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

import os
import uuid
import json
import base64
import aiohttp
import logging
from fastapi import UploadFile
from fastapi.responses import JSONResponse
import subprocess
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Configure logging
logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

async def handle_speech_to_text(file: UploadFile):
    logger.info(f"Starting speech to text conversion for file: {file.filename}")
    
    if file is None:
        logger.error("No file provided")
        return JSONResponse(
            status_code=400,
            content={"error": "No file provided."}
        )
    
    # Get API key from environment
    api_key = os.getenv('GOOGLE_API_KEY')
    if not api_key:
        logger.error("GOOGLE_API_KEY not found in environment variables")
        return JSONResponse(
            status_code=500,
            content={"error": "Google API key not configured"}
        )

    # Create data directory if it doesn't exist
    os.makedirs("data", exist_ok=True)
    
    # Save the uploaded file temporarily
    webm_path = os.path.join("data", f"temp_{uuid.uuid4().hex}.webm")
    wav_path = os.path.join("data", f"temp_{uuid.uuid4().hex}.wav")
    
    try:
        # Read the uploaded file
        logger.debug("Reading uploaded file")
        content = await file.read()
        with open(webm_path, "wb") as f:
            f.write(content)
        logger.debug(f"Saved WebM file to {webm_path}")
        
        # Convert the WebM file to WAV format
        logger.debug("Converting WebM to WAV")
        try:
            result = subprocess.run([
                'ffmpeg', '-i', webm_path, 
                '-acodec', 'pcm_s16le', 
                '-ar', '16000', 
                '-ac', '1', 
                wav_path
            ], check=True, capture_output=True, text=True)
            logger.debug(f"FFmpeg output: {result.stdout}")
            logger.debug(f"FFmpeg errors: {result.stderr}")
        except subprocess.CalledProcessError as e:
            logger.error(f"FFmpeg conversion failed: {str(e)}")
            logger.error(f"FFmpeg stderr: {e.stderr}")
            return JSONResponse(
                status_code=500,
                content={"error": f"FFmpeg conversion failed: {e.stderr}"}
            )

        # Verify WAV file exists and has content
        if not os.path.exists(wav_path):
            logger.error("WAV file was not created")
            return JSONResponse(
                status_code=500,
                content={"error": "WAV file conversion failed"}
            )
            
        wav_size = os.path.getsize(wav_path)
        logger.debug(f"WAV file size: {wav_size} bytes")
        
        if wav_size == 0:
            logger.error("WAV file is empty")
            return JSONResponse(
                status_code=500,
                content={"error": "Converted WAV file is empty"}
            )

        # Read the WAV file and encode it to base64
        logger.debug("Reading and encoding WAV file")
        with open(wav_path, "rb") as audio_file:
            audio_content = base64.b64encode(audio_file.read()).decode('utf-8')

        # Prepare the request payload
        payload = {
            "config": {
                "encoding": "LINEAR16",
                "sampleRateHertz": 16000,
                "languageCode": "en-US",
                "enableAutomaticPunctuation": True
            },
            "audio": {
                "content": audio_content
            }
        }

        # Make request to Google Cloud Speech-to-Text API
        logger.debug("Making request to Google Cloud API")
        async with aiohttp.ClientSession() as session:
            url = f"https://speech.googleapis.com/v1/speech:recognize?key={api_key}"
            async with session.post(url, json=payload) as response:
                response_text = await response.text()
                logger.debug(f"API Response status: {response.status}")
                logger.debug(f"API Response: {response_text}")
                
                if response.status != 200:
                    logger.error(f"API request failed with status {response.status}: {response_text}")
                    return JSONResponse(
                        status_code=response.status,
                        content={"error": f"API request failed: {response_text}"}
                    )
                
                result = json.loads(response_text)
                
                # Extract the transcribed text
                transcript = ""
                if "results" in result:
                    for result_item in result["results"]:
                        if "alternatives" in result_item and result_item["alternatives"]:
                            transcript += result_item["alternatives"][0]["transcript"]
                
                logger.info("Successfully transcribed audio")
                return {"status": "success", "text": transcript}

    except Exception as e:
        logger.exception("Unexpected error occurred")
        return JSONResponse(
            status_code=500,
            content={"error": f"An error occurred during transcription: {str(e)}"}
        )
    
    finally:
        # Clean up temporary files
        logger.debug("Cleaning up temporary files")
        if os.path.exists(webm_path):
            os.remove(webm_path)
            logger.debug(f"Removed {webm_path}")
        if os.path.exists(wav_path):
            os.remove(wav_path)
            logger.debug(f"Removed {wav_path}")

def initialize_global_vectorstore():
    global global_vectorstore
    with vectorstore_lock:
        if global_vectorstore is not None:
            return True, "[SYSTEM MESSAGE] Vectorstore is already initialized."

        pdf_paths = [
            os.path.join('data', "Ilesh Sir (IK) - Words.pdf"),
            os.path.join('data', "UBIK SOLUTION.pdf"),
            os.path.join('data', "illesh3.pdf"),
            os.path.join('data', "website-data-ik.pdf"),
            os.path.join('data', "prods1.pdf")
        ]

        combined_text = ""
        for path in pdf_paths:
            combined_text += get_pdf_text(path) + " "

        if not combined_text.strip():
            return False, "No text could be extracted from the PDFs."

        text_chunks = get_text_chunks(combined_text)
        
        # Initialize the embedding model
        model_kwargs = {'device': 'cpu'}
        encode_kwargs = {'normalize_embeddings': True}
        embeddings = HuggingFaceEmbeddings(
            model_name=EMBEDDING_MODEL,
            model_kwargs=model_kwargs,
            encode_kwargs=encode_kwargs
        )
        
        global_vectorstore = FAISS.from_texts(
            texts=text_chunks,
            embedding=embeddings
        )
        logger.info("Vectorstore has been created with SentenceTransformers embeddings.")
    return True, "[SYSTEM MESSAGE] Vectorstore was created successfully."

def handle_userinput(user_question: str, user_id: str):
    user_state = get_user_state(user_id)
    conversation_chain = user_state['chain']
    if not conversation_chain:
        return None

    # Step 1: Define the refinement template
    def refine_input(input_text):
        refinement_prompt = f"""
        The user provided the following input: "{input_text}"

        Context: The input may contain minor errors or variations due to inaccuracies in communication or understanding. something as huge as ehiglo considered as ethical law or igloo or tigloo so be lineant
        Your task is to:
        1. Interpret the user's intent.
        2. Refine the input to make it coherent and meaningful.
        3. Provide the refined version.

        Refined Input:
        """
        return conversation_chain({'question': refinement_prompt})['answer'].strip()

    # Step 2: Generate the answer using the refined input
    def generate_answer(refined_input):
        answer_prompt = f"""
        User's Refined Input: "{refined_input}"

        Context: Respond to the user's query based on the refined input in a clear, concise, and contextually relevant manner.

        Answer:
        """
        return conversation_chain({'question': answer_prompt})['answer'].strip()

    # Step 3: Refine the user input
    refined_question = refine_input(user_question)
    log_event("RefinedInput", f"Refined Question: {refined_question}", user_id=user_id)

    # Step 4: Generate the final answer
    answer = generate_answer(refined_question)

    # Step 5: Log events and update history
    user_state['history'].append((user_question, answer))
    log_event("PromptSentToGPT", f"Original Prompt: {user_question}, Refined: {refined_question}", user_id=user_id)
    log_event("UserQuestion", f"Q: {user_question}", user_id=user_id)
    log_event("AIAnswer", f"A: {answer}", user_id=user_id)

    return {'text': answer}

def create_or_refresh_user_chain(user_id: str):
    user_state = get_user_state(user_id)
    if user_state['chain'] is None:
        if global_vectorstore is None:
            return False, "Global vectorstore is not initialized."

        chat_llm = ChatOpenAI(model=MODEL_NAME, temperature=0.9)

        # Enhanced condense template that preserves potentially unclear words
        condense_template = """Given the conversation and follow-up question, rephrase it as a standalone question.
        Keep any unclear words in their original form for context matching.

        Chat History:
        {chat_history}
        Follow Up Input: {question}
        Standalone question:"""
        
        CONDENSE_QUESTION_PROMPT = PromptTemplate.from_template(condense_template)

        # Enhanced QA template with context-aware clarification logic
        qa_template = f"""
        {user_state['system_message']}

        You are trained specifically about Ilesh Sir and UBIK Solutions. Follow this exact decision tree when responding:

        1. First, check if the question has any relevant context in the provided knowledge base:
           - If you find context even with misspelled words → Proceed with answering

        2. If no direct context match but the question seems relevant:
           - If unclear word might affect the answer → Ask for spelling playfully
           - Example: "Oh! I'd love to help with that! Could you spell out [unclear word]? Just want to make sure I give you the perfect answer! 😊"
           
        3. If the question seems relevant but unclear word isn't crucial:
           - Proceed with answering based on best interpretation
           - Example: Small grammatical errors or common word variations

        4. If the question is completely irrelevant and u get no context along:
           - Politely respond: "I'm specifically trained to help with questions about UBIK Solutions. This seems outside my expertise!"

        Context: {context}
        
        Question: {question}
        
        Answer: """
        
        QA_PROMPT = PromptTemplate(
            template=qa_template,
            input_variables=["context", "question"]
        )

        user_state['chain'] = ConversationalRetrievalChain.from_llm(
            llm=chat_llm,
            retriever=global_vectorstore.as_retriever(),
            memory=user_state['memory'],
            condense_question_prompt=CONDENSE_QUESTION_PROMPT,
            combine_docs_chain_kwargs={'prompt': QA_PROMPT}
        )
        
        logger.info(f"New conversation chain created for user {user_id} with context-aware prompt template")
        return True, "Conversation chain created with intelligent spelling clarification."
    else:
        return True, "Conversation chain already exists."
    
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
    FORWARD_ENDPOINT = os.getenv("FORWARD_ENDPOINT", "https://d07f-157-119-42-46.ngrok-free.app/receive")

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


# You can remove or comment out the StaticFiles mounting since we're not storing files anymore
# app.mount("/data", StaticFiles(directory="data"), name="data")
#pip install sentence-transformers transformers torch
# Command to run:
# uvicorn server3:app --host 0.0.0.0 --port 8000 --reload
