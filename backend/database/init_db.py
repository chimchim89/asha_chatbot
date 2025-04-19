import aiosqlite
from contextlib import asynccontextmanager

@asynccontextmanager
async def get_db():
    async with aiosqlite.connect("asha_chatbot.db") as conn:
        yield conn

async def init_db():
    async with aiosqlite.connect("asha_chatbot.db") as conn:
        await conn.execute("""
            CREATE TABLE IF NOT EXISTS users (
                user_id TEXT PRIMARY KEY,
                skills TEXT,
                preferences TEXT
            )
        """)
        await conn.commit()
