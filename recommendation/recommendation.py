import asyncio
from database import db_repo as repo
# from geoalchemy2.shape import to_shape
from geopy.distance import geodesic
from database.db_models import UserRecommendationParams, QuizEditionRecommendationParams, UserTopRecommendations


def compute_match(
    user: "UserRecommendationParams",
    edition: "QuizEditionRecommendationParams",
    weights: dict = None
) -> float:

    if weights is None:
        weights = {
            "rating": 5,
            "duration": 2,
            "team_size": 1,
            "categories": 5,
            "hosts": 2,
            "day_of_week": 1,
            "time_of_edition": 1,
            "location": 3
        }

    score = 0.0

    max_rating_diff = 200
    total_weight = sum(weights.values())

    elo_diff = abs(user.rating - edition.rating)
    rating_score = 1 - min(elo_diff / max_rating_diff, 1.0)
    score += weights["rating"] * rating_score

    duration_diff = abs(user.duration - edition.duration)
    duration_score = 1 - min(duration_diff, 1.0)
    score += weights["duration"] * duration_score

    team_diff = abs(user.team_size * user.number_of_teams - edition.team_size * edition.number_of_teams)
    team_score = 1 - min(team_diff, 1.0)
    score += weights["team_size"] * team_score

    user_categories = dict(user.categories)
    edition_categories = set(edition.category_ids)
    total_sum = sum(user_categories.values())
    matching_sum = sum(
        user_categories[str(cat_id)]
        for cat_id in edition_categories
        if str(cat_id) in user_categories
    )
    category_score = matching_sum / total_sum
    score += weights["categories"] * category_score

    # --- Host match ---
    user_hosts = dict(user.hosts)
    host_total_sum = sum(user_hosts.values())
    host_score = user_hosts.get(str(edition.host_id), 0) / host_total_sum
    score += weights["hosts"] * host_score

    # --- Day of the week match ---
    user_day_of_week = list(user.day_of_week)
    day_of_week_total_sum = sum(user_day_of_week)
    day_of_week_score = user_day_of_week[edition.day_of_the_week] / day_of_week_total_sum
    score += weights["day_of_week"] * day_of_week_score

    time_diff = abs(user.time_of_edition - edition.time_of_edition)
    time_score = 1 - min(time_diff / 12.0, 1.0)
    score += weights["time_of_edition"] * time_score

    if user.last_known_location and edition.location:
        distance_m = geodesic(user.last_known_location, edition.location).meters
        location_score = max(0, min(0, 1 - distance_m / 20000))
    else:
        location_score = 0

    score += weights["location"] * location_score

    final_score = score / total_weight
    return final_score


async def get_recommendations(edition_id):
    edition_params = await repo.get_edition_recommendations_params(edition_id)
    edition_time = await repo.get_edition_time(edition_id)
    user_params = await repo.get_user_recommendations_params(edition_params.location)

    for user in user_params:
        score = compute_match(user, edition_params)
        await repo.add_user_top_recommendation(UserTopRecommendations(user_id=user.user_id, edition_id=edition_params.edition_id,  edition_timestamp=edition_time, match=score))
        print(score)

if __name__ == "__main__":
    asyncio.run(get_recommendations(edition_id=1))
