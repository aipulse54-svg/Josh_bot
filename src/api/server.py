"""FastAPI server for LLM Assistant."""
import logging
from typing import Optional
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from src.model.llm import LLMAssistant
from src.inference.assistant import CodeMathAssistant

logger = logging.getLogger(__name__)

# Initialize FastAPI app
app = FastAPI(
    title="LLM Assistant API",
    description="AI Assistant for Code Generation and Math Solving",
    version="1.0.0",
)

# Request/Response models
class CodeGenerationRequest(BaseModel):
    prompt: str = Field(..., description="Code generation prompt")
    language: str = Field(default="python", description="Programming language")
    max_length: int = Field(default=512, description="Maximum length of generated code")


class CodeGenerationResponse(BaseModel):
    prompt: str
    language: str
    code: str
    success: bool


class MathProblemRequest(BaseModel):
    problem: str = Field(..., description="Mathematical problem")


class MathProblemResponse(BaseModel):
    problem: str
    solutions: list
    method: str
    success: bool


class QuestionRequest(BaseModel):
    question: str = Field(..., description="Question to answer")
    context: Optional[str] = Field(default=None, description="Optional context")


class QuestionResponse(BaseModel):
    question: str
    answer: str
    context: Optional[str]
    success: bool


class CodeExplanationRequest(BaseModel):
    code: str = Field(..., description="Code to explain")


class CodeExplanationResponse(BaseModel):
    code: str
    explanation: str
    success: bool


# Global variables (will be initialized)
_llm_model = None
_assistant = None


def init_model(model_name: str = "gpt2"):
    """Initialize the LLM model."""
    global _llm_model, _assistant
    logger.info(f"Initializing model: {model_name}")
    _llm_model = LLMAssistant(model_name=model_name)
    _assistant = CodeMathAssistant(_llm_model)
    logger.info("Model initialized successfully")


@app.on_event("startup")
async def startup_event():
    """Initialize model on startup."""
    init_model()


@app.get("/health")
async def health_check():
    """Health check endpoint."""
    return {
        "status": "healthy",
        "model_loaded": _llm_model is not None,
    }


@app.post("/generate-code", response_model=CodeGenerationResponse)
async def generate_code(request: CodeGenerationRequest):
    """Generate code based on prompt."""
    try:
        result = _assistant.generate_code(
            prompt=request.prompt,
            language=request.language,
            max_length=request.max_length,
        )
        return CodeGenerationResponse(**result)
    except Exception as e:
        logger.error(f"Error in generate_code: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/solve-math", response_model=MathProblemResponse)
async def solve_math(request: MathProblemRequest):
    """Solve a mathematical problem."""
    try:
        result = _assistant.solve_math_problem(problem=request.problem)
        return MathProblemResponse(**result)
    except Exception as e:
        logger.error(f"Error in solve_math: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/answer-question", response_model=QuestionResponse)
async def answer_question(request: QuestionRequest):
    """Answer a question."""
    try:
        result = _assistant.answer_question(
            question=request.question,
            context=request.context,
        )
        return QuestionResponse(**result)
    except Exception as e:
        logger.error(f"Error in answer_question: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/explain-code", response_model=CodeExplanationResponse)
async def explain_code(request: CodeExplanationRequest):
    """Explain a piece of code."""
    try:
        result = _assistant.explain_code(code=request.code)
        return CodeExplanationResponse(**result)
    except Exception as e:
        logger.error(f"Error in explain_code: {e}")
        raise HTTPException(status_code=500, detail=str(e))


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
