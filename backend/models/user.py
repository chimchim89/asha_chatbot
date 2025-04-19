from pydantic import BaseModel

class User(BaseModel):
    user_id: str = None
    name: str = None
    skills: str = None
    background: str = None
    preferences: str = None
