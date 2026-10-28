import os
import asyncio

from dotenv import load_dotenv

from sqlalchemy.ext.asyncio import (
    AsyncSession,
    async_sessionmaker,
    create_async_engine,
)
from sqlalchemy.orm import DeclarativeBase

load_dotenv()

DB_URL = os.getenv("DATABASE_URL")

if not DB_URL:
    raise RuntimeError("DATABASE_URL is not set")

engine = create_async_engine(
    DB_URL,
    pool_pre_ping=True,
    pool_size=10,
    max_overflow=20,
)

sessionLocal = async_sessionmaker(
    engine,
    class_=AsyncSession,
    expire_on_commit=False,
)

class Base(DeclarativeBase):
    pass

async def test():
    async with engine.connect() as conn:
        print("✅ Successfully connected via db.py!")

# async def get_db():
#     async with sessionLocal() as session:
#         yield session

async def get_db():
    async with sessionLocal() as session:
        try:
            yield session
            await session.commit()  # Automatically commit changes if successful
        except Exception:
            await session.rollback() # Rollback if something breaks
            raise

async def close_db():
    await engine.dispose()

if __name__ == "__main__":
    asyncio.run(test())