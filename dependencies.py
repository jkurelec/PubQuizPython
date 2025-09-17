from db.db import async_session
from sqlalchemy.ext.asyncio import AsyncSession


async def get_db_session() -> AsyncSession:
    async with async_session() as session:
        yield session

