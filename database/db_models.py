from sqlalchemy import Column, Integer, Float, TIMESTAMP, ForeignKey, String
from sqlalchemy.dialects.postgresql import ARRAY, JSONB
# from geoalchemy2 import Geometry
from sqlalchemy.orm import declarative_base
from sqlalchemy.sql import func

Base = declarative_base()


class QuizEditionRecommendationParams(Base):
    __tablename__ = "quiz_edition_recommendation_params"
    __table_args__ = {"schema": "public"}

    edition_id = Column(Integer, ForeignKey("public.quiz_edition.id"), primary_key=True)
    rating = Column(Integer, nullable=False)
    duration = Column(Float, nullable=False)
    category_ids = Column(ARRAY(Integer), nullable=False)
    host_id = Column(Integer, ForeignKey("public.user.id"), nullable=False)
    number_of_teams = Column(Float, nullable=False)
    team_size = Column(Float, nullable=False)
    day_of_the_week = Column(Integer, nullable=False)
    time_of_edition = Column(Integer, nullable=False)
    location = Column(String, nullable=False)


class UserFeedback(Base):
    __tablename__ = "user_feedback"
    __table_args__ = {"schema": "public"}

    user_id = Column(Integer, primary_key=True)
    edition_id = Column(Integer, primary_key=True)
    general_rating = Column(Float, nullable=False)
    host_rating = Column(Float, nullable=False)
    difficulty_rating = Column(Float, nullable=False)
    duration_rating = Column(Float, nullable=False)
    number_of_people_rating = Column(Float, nullable=False)
    timestamp = Column(TIMESTAMP(timezone=True), nullable=False)
    user_rating = Column(Integer, nullable=False)


class UserRecommendationParams(Base):
    __tablename__ = "user_recommendation_params"
    __table_args__ = {"schema": "public"}

    user_id = Column(Integer, ForeignKey("public.user.id"), primary_key=True)
    rating = Column(Integer, nullable=False)
    duration = Column(Float, nullable=False)
    number_of_teams = Column(Float, nullable=False)
    team_size = Column(Float, nullable=False)
    edition_count = Column(Integer, nullable=False)
    hosts = Column(JSONB, nullable=False)
    categories = Column(JSONB, nullable=False)
    time_of_edition = Column(Float, nullable=False)
    day_of_week = Column(ARRAY(Integer), nullable=False)
    last_known_location = Column(String)
    last_login = Column(TIMESTAMP(timezone=True), nullable=False)


class UserTopRecommendations(Base):
    __tablename__ = "user_top_recommendations"
    __table_args__ = {"schema": "public"}

    user_id = Column(Integer, primary_key=True)
    edition_id = Column(Integer, primary_key=True)
    edition_timestamp = Column(TIMESTAMP(timezone=True), nullable=False)
    created_at = Column(TIMESTAMP(timezone=True), server_default=func.now())
    match = Column(Float, nullable=True)
