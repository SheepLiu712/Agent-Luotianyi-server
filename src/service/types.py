
from pydantic import BaseModel
class ChatRequest(BaseModel):
    text: str
    username: str
    token: str

class HistoryRequest(BaseModel):
    username: str
    token: str
    count: int = 10
    end_index: int = -1

class ChatResponse(BaseModel):
    text: str
    audio: str | None = None
    expression: str | None = None

class LoginRequest(BaseModel):
    username: str
    password: str
    request_token: bool = False

class RegisterRequest(BaseModel):
    username: str
    password: str
    invite_code: str

class AutoLoginRequest(BaseModel):
    username: str
    token: str

