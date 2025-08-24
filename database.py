from sqlmodel import create_engine, Session
import os

DATABASE_URL = os.getenv("DATABASE_URL", "postgresql://postgres:123456@localhost:5432/questgendb")

engine = create_engine(DATABASE_URL)

def get_db():
    with Session(engine) as session:
        yield session
