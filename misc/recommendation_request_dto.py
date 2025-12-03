from pydantic import BaseModel


class recommendation_request_dto(BaseModel):
    id: int