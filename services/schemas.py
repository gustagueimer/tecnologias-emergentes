from pydantic import BaseModel, Field


class ChatRequest(BaseModel):
    message: str = Field(..., min_length=1, max_length=2000)
    limit: int = Field(default=10, ge=1, le=100)


class ChatResponse(BaseModel):
    agent_name: str
    agent_role: str
    events_analyzed: int
    summary: str