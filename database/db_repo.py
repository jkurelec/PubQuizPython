import math
from datetime import datetime, timezone, timedelta

from sqlalchemy import select
from database.db_models import UserTopRecommendations, UserRecommendationParams, QuizEditionRecommendationParams
from database.db import async_session
from sqlalchemy.ext.asyncio import AsyncSession
from contextlib import asynccontextmanager
from sqlalchemy import func, text


@asynccontextmanager
async def get_db_session() -> AsyncSession:
    async with async_session() as session:
        yield session


async def get_user_recommendations(user_id):
    async with get_db_session() as session:
        result = await session.execute(
            select(UserTopRecommendations)
            .where(UserTopRecommendations.user_id == user_id)
        )
        recommendations = result.scalars().all()
        return [r.__dict__ for r in recommendations]


async def get_user_recommendations_params(edition_location, days_back=7, km_radius=20):
    cutoff = datetime.now(timezone.utc) - timedelta(days=days_back)
    lon, lat = edition_location
    R = 6371

    async with get_db_session() as session:
        result = await session.execute(
            select(UserRecommendationParams)
            .where(UserRecommendationParams.last_login >= cutoff)
            .where(
                text(
                    f"""
                    (
                        {R} * acos(
                            cos(radians({lat})) * cos(radians((last_known_location)[1])) *
                            cos(radians((last_known_location)[0]) - radians({lon})) +
                            sin(radians({lat})) * sin(radians((last_known_location)[1]))
                        )
                    ) <= {km_radius}
                    """
                )
            )
        )
        users = result.scalars().all()

    return users


async def get_edition_recommendations_params(edition_id):
    async with get_db_session() as session:
        result = await session.execute(
            select(QuizEditionRecommendationParams)
            .where(QuizEditionRecommendationParams.edition_id == edition_id)
        )
        return result.scalars().first()


async def get_edition_time(edition_id):
    async with get_db_session() as session:
        result = await session.execute(
            text("SELECT time FROM public.quiz_edition WHERE id = :edition_id"),
            {"edition_id": edition_id}
        )
        row = result.first()
        return row[0] if row else None


async def add_user_top_recommendation(recommendation: UserTopRecommendations):
    async with get_db_session() as session:
        session.add(recommendation)
        await session.commit()


def parse_point(point_str: str):
    x, y = point_str.strip("()").split(",")
    return float(x), float(y)