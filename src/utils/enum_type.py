from enum import Enum

class ContextType(str, Enum):
    TEXT = "text",
    SING = "sing"
    CMD = "cmd",
    PIC = "pic",

class ConversationSource(str, Enum):
    USER = "user",
    AGENT = "agent",
    SYSTEM = "system",

