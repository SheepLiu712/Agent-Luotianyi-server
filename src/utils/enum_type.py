from enum import Enum

class ContextType(str, Enum):
    TEXT = "text"
    SING = "sing"
    CMD = "cmd"
    PICTURE = "picture"


class ConversationSource(str, Enum):
    USER = "user",
    AGENT = "agent",
    SYSTEM = "system",

