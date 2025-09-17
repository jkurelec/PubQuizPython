from fastapi import FastAPI, Depends
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select
from dependencies import get_db_session
from db.db_models import UserTopRecommendations


app = FastAPI()


@app.get("/recommendations/{user_id}")
async def get_recommendations(user_id: int, session: AsyncSession = Depends(get_db_session)):
    query = select(UserTopRecommendations).where(UserTopRecommendations.user_id == user_id)
    result = await session.execute(query)
    recommendations = result.scalars().all()
    return [r.__dict__ for r in recommendations]


@app.get("/")
async def root():
    return {"message": "Hello World"}


@app.get("/hello/{name}")
async def say_hello(name: str):
    return {"message": f"Hello {name}"}
