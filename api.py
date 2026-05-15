from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Optional, List, Dict, Any
import uuid

from langchain_core.runnables import RunnableConfig
from langgraph.types import Command
from graph import lodestone_graph   # use the singleton, not build_graph()

app = FastAPI(title="LODESTONE API", description="Business Research Assistant API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

sessions: Dict[str, Any] = {}

class ChatRequest(BaseModel):
    message: str
    session_id: Optional[str] = None

class ChatResponse(BaseModel):
    session_id: str
    response: str
    follow_ups: List[str] = []
    status: str = "success"


def extract_response(final_state: dict):
    final_output = final_state.get("final_response", "") or ""
    response_text = final_output
    follow_ups = final_state.get("suggested_followups", [])

    if not follow_ups and "Suggested follow-ups:" in final_output:
        parts = final_output.split("Suggested follow-ups:")
        response_text = parts[0].strip()
        for line in parts[1].strip().split('\n'):
            line = line.strip()
            if line and line[0].isdigit() and '. ' in line:
                follow_ups.append(line.split('. ', 1)[1])
            elif line.startswith('*'):
                follow_ups.append(line[1:].strip())

    return response_text or "I could not generate a response.", follow_ups


@app.post("/api/chat", response_model=ChatResponse)
async def chat_endpoint(request: ChatRequest):
    session_id = request.session_id

    if not session_id or session_id not in sessions:
        session_id = str(uuid.uuid4())
        sessions[session_id] = {"interrupted": False}

    config: RunnableConfig = {"configurable": {"thread_id": session_id}}

    try:
        # Check if this thread is currently paused at an interrupt
        current_state = lodestone_graph.get_state(config)
        is_interrupted = bool(current_state.next) and "human_feedback" in current_state.next

        if is_interrupted:
            # Resume the paused graph with the user's clarification text
            final_state = lodestone_graph.invoke(
                Command(resume=request.message),
                config=config
            )
        else:
            # Fresh query
            final_state = lodestone_graph.invoke(
                {"raw_input": request.message},
                config=config
            )

        # Check if we ended up at another interrupt (clarification needed)
        new_state = lodestone_graph.get_state(config)
        if bool(new_state.next) and "human_feedback" in new_state.next:
            question = final_state.get("clarification_question", "Could you clarify your question?")
            return ChatResponse(
                session_id=session_id,
                response=question,
                follow_ups=[],
                status="needs_clarification"
            )

        response_text, follow_ups = extract_response(final_state)
        return ChatResponse(
            session_id=session_id,
            response=response_text,
            follow_ups=follow_ups,
            status="success"
        )

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/sessions")
async def get_sessions():
    return {"sessions": list(sessions.keys())}