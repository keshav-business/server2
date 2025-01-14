from fastapi import FastAPI, HTTPException, Depends, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List, Optional, Dict
import uvicorn
import uuid
import os
from datetime import datetime, timedelta
from fastapi.security import APIKeyHeader
from starlette.requests import Request
from dotenv import load_dotenv
from PyPDF2 import PdfReader
from langchain.text_splitter import CharacterTextSplitter
from langchain_community.embeddings import OpenAIEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_community.chat_models import ChatOpenAI
from langchain.memory import ConversationBufferMemory
from langchain.chains import ConversationalRetrievalChain
import openai
import tempfile
import logging
from openai import OpenAI
# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

load_dotenv()

# Initialize OpenAI API key
openai.api_key = os.getenv('OPENAI_API_KEY')

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class SessionManager:
    def __init__(self, session_timeout_minutes: int = 30):
        self.sessions: Dict[str, dict] = {}
        self.session_timeout = timedelta(minutes=session_timeout_minutes)
        
    def create_session(self) -> str:
        session_id = str(uuid.uuid4())
        self.sessions[session_id] = {
            'created_at': datetime.now(),
            'last_accessed': datetime.now(),
            'chain': None,
            'memory': ConversationBufferMemory(memory_key='chat_history', return_messages=True),
            'active_quizzes': {}
        }
        return session_id
    
    def get_session(self, session_id: str) -> Optional[dict]:
        session = self.sessions.get(session_id)
        if session:
            if datetime.now() - session['last_accessed'] > self.session_timeout:
                del self.sessions[session_id]
                return None
            session['last_accessed'] = datetime.now()
        return session
    
    def cleanup_expired_sessions(self):
        current_time = datetime.now()
        expired_sessions = [
            sid for sid, session in self.sessions.items()
            if current_time - session['last_accessed'] > self.session_timeout
        ]
        for sid in expired_sessions:
            del self.sessions[sid]

session_manager = SessionManager()

API_KEY_HEADER = APIKeyHeader(name="X-Session-ID", auto_error=False)

async def get_session(session_id: Optional[str] = Depends(API_KEY_HEADER)) -> dict:
    if not session_id:
        session_id = session_manager.create_session()
    
    session = session_manager.get_session(session_id)
    if not session:
        session_id = session_manager.create_session()
        session = session_manager.get_session(session_id)
    
    return {'session_id': session_id, 'data': session}

class QuizAnswer(BaseModel):
    question_id: str
    user_answer: str

class Question(BaseModel):
    id: str
    question: str
    ideal_answer: Optional[str] = None

SYSTEM_MESSAGE = os.getenv('OPENAI_SYSTEM_MESSAGE', 'You are a helpful AI assistant specializing in document analysis.')
CHUNK_SIZE = int(os.getenv('CHUNK_SIZE', 1000))
CHUNK_OVERLAP = int(os.getenv('CHUNK_OVERLAP', 200))
MODEL_NAME = os.getenv('MODEL_NAME', 'gpt-4-turbo-preview')
PDF_PATHS = [
    os.path.join('data', "Ilesh Sir (IK) - Words.pdf"),
    os.path.join('data', "UBIK SOLUTION.pdf"),
    os.path.join('data', "illesh3.pdf"),
    os.path.join('data', "website-data-ik.pdf"),
]

global_vectorstore = None

def get_pdf_text(pdf_path: str) -> str:
    reader = PdfReader(pdf_path)
    text = ""
    for page in reader.pages:
        text += page.extract_text() or ""
    return text

def get_text_chunks(text: str) -> List[str]:
    text_splitter = CharacterTextSplitter(
        separator="\n",
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
        length_function=len
    )
    return text_splitter.split_text(text)

def initialize_vectorstore():
    global global_vectorstore
    if global_vectorstore is not None:
        return

    combined_text = ""
    for path in PDF_PATHS:
        combined_text += get_pdf_text(path) + " "
    
    text_chunks = get_text_chunks(combined_text)
    global_vectorstore = FAISS.from_texts(
        texts=text_chunks,
        embedding=OpenAIEmbeddings()
    )

def create_chain(session_data: dict):
    if global_vectorstore is None:
        raise Exception("Vectorstore not initialized")

    chat_llm = ChatOpenAI(
        model=MODEL_NAME,
        temperature=0.7
    )

    session_data['chain'] = ConversationalRetrievalChain.from_llm(
        llm=chat_llm,
        retriever=global_vectorstore.as_retriever(),
        memory=session_data['memory']
    )

def generate_questions(chain, num_questions: int = 5) -> List[Question]:
    prompt = f"{SYSTEM_MESSAGE}\n\nGenerate {num_questions} questions about Ubik Solutions and their products/services. Make questions that test understanding of key concepts and details."
    response = chain({'question': prompt})
    answer_text = response.get('answer', '')
    
    # Split the response into individual questions
    question_lines = [line.strip() for line in answer_text.split('\n') if line.strip()]
    questions = []
    
    for line in question_lines:
        if line:
            question_id = str(uuid.uuid4())
            questions.append(Question(
                id=question_id,
                question=line
            ))
    
    return questions[:num_questions]

@app.on_event("startup")
async def startup_event():
    initialize_vectorstore()



# Initialize the client once at the module level
client = OpenAI(api_key=os.getenv('OPENAI_API_KEY'))

@app.post("/transcribe-audio")
async def transcribe_audio(
    audio_file: UploadFile = File(...),
    session: dict = Depends(get_session)
):
    try:
        # Create a temporary file to store the uploaded audio
        with tempfile.NamedTemporaryFile(delete=False, suffix='.webm') as temp_audio:
            # Write the uploaded file content to temporary file
            content = await audio_file.read()
            temp_audio.write(content)
            temp_audio.flush()
            
            # Open the file and send to Whisper API using new client format
            with open(temp_audio.name, 'rb') as audio:
                transcript = client.audio.transcriptions.create(
                    model="whisper-1",
                    file=audio
                )
        
        # Clean up temporary file
        os.unlink(temp_audio.name)
        
        return {"text": transcript.text}
        
    except Exception as e:
        logger.error(f"Error in audio transcription: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail="Error processing audio transcription"
        )
@app.post("/quiz/start")
async def start_quiz(session: dict = Depends(get_session)):
    session_data = session['data']
    if not session_data.get('chain'):
        create_chain(session_data)
    
    quiz_id = str(uuid.uuid4())
    questions = generate_questions(session_data['chain'])
    
    session_data['active_quizzes'][quiz_id] = {
        "questions": questions,
        "answers": {},
        "current_question": 0
    }
    
    return {
        "session_id": session['session_id'],
        "quiz_id": quiz_id,
        "questions": questions
    }

@app.post("/quiz/answer")
async def submit_answer(request: Request):
    try:
        # Get the raw JSON data from request
        data = await request.json()
        
        logger.info(f"Received answer data: {data}")
        
        # Return success response with minimal processing
        return {
            "status": "success",
            "quiz_complete": False,  # You can modify this based on your needs
            "received_data": data
        }
    except Exception as e:
        logger.error(f"Error processing answer: {str(e)}")
        return {
            "status": "error",
            "message": str(e)
        }
@app.get("/quiz/{quiz_id}/result")
async def get_quiz_result(quiz_id: str, session: dict = Depends(get_session)):
    session_data = session['data']
    quiz = session_data['active_quizzes'].get(quiz_id)
    
    if not quiz:
        raise HTTPException(status_code=404, detail="Quiz not found")
    
    # Get the chain to evaluate answers
    if not session_data.get('chain'):
        create_chain(session_data)
    
    evaluated_answers = []
    
    for question in quiz["questions"]:
        user_answer = quiz["answers"].get(question.id, "No answer provided")
        # You could add AI evaluation of answers here if desired
        evaluated_answers.append({
            "question": question.question,
            "user_answer": user_answer,
        })
    
    return {
        "quiz_id": quiz_id,
        "answers": evaluated_answers,
        "total_questions": len(quiz["questions"]),
        "answered_questions": len(quiz["answers"])
    }

@app.get("/debug/session")
async def debug_session(session: dict = Depends(get_session)):
    """Debug endpoint to view current session state"""
    session_data = session['data']
    return {
        "session_id": session['session_id'],
        "active_quizzes": {
            quiz_id: {
                "current_question": quiz["current_question"],
                "total_questions": len(quiz["questions"]),
                "answered_questions": list(quiz["answers"].keys()),
            }
            for quiz_id, quiz in session_data.get('active_quizzes', {}).items()
        }
    }

if __name__ == "__main__":
    uvicorn.run("server2:app", host="0.0.0.0", port=9000, reload=True)
