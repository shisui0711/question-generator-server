from typing import Optional
from sqlmodel import Field, SQLModel
from datetime import datetime


class UserBase(SQLModel):
    username: str = Field(index=True)
    email: str = Field(unique=True, index=True)
    full_name: Optional[str] = None


class User(UserBase, table=True):
    id: Optional[int] = Field(default=None, primary_key=True)
    hashed_password: str
    is_active: bool = Field(default=True)
    role: str = Field(default="user")
    created_at: datetime = Field(default_factory=datetime.utcnow, nullable=False)


class UserCreate(UserBase):
    password: str

class UserLogin(SQLModel):
    username: str
    password: str

class UserResponse(UserBase):
    id: int
    is_active: bool
    role: str
    created_at: datetime
