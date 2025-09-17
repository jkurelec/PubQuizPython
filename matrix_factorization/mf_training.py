import numpy as np
from datetime import datetime


def normalize_dict(d):
    total = sum(d.values())

    return {k: v / total for k, v in d.items()} if total > 0 else {k: 0 for k in d}


def clamp_and_normalize(value, min_val=-200, max_val=200):
    clamped = max(min(value, max_val), min_val)

    return (clamped - min_val) / (max_val - min_val)


def compute_user_feedback(user_params, edition_params):
    difficulty_delta = user_params['rating'] - edition_params['rating']
    difficulty_normalized = clamp_and_normalize(difficulty_delta)
    difficulty_rating = 1 if difficulty_delta > 20 else -1 if difficulty_delta < -20 else 0

    # Host rating
    normalized_hosts = normalize_dict(user_params['hosts'])
    host_rating = normalized_hosts.get(edition_params['host_id'], 0)

    # Duration rating
    duration_delta = edition_params['duration'] - user_params['duration']
    duration_rating = 1 if duration_delta > 0.2 else -1 if duration_delta < -0.2 else 0

    # Number of people rating
    edition_people = (edition_params['number_of_teams'] + edition_params['team_size']) / 2
    user_people = (user_params['number_of_teams'] + user_params['team_size']) / 2
    people_delta = edition_people - user_people
    number_of_people_rating = 1 if people_delta > 0.2 else -1 if people_delta < -0.2 else 0

    # General rating components
    time_score = 1 - abs(edition_params['time_of_edition'] - user_params['time_of_edition'])

    day_distribution = user_params['day_of_week']
    day_total = sum(day_distribution)
    day_score = (day_distribution[edition_params['day_of_the_week']] / day_total) if day_total > 0 else 0

    normalized_categories = normalize_dict(user_params['categories'])
    category_score = sum(normalized_categories.get(cid, 0) for cid in edition_params['category_ids'])

    general_rating = np.mean([time_score, day_score, category_score])

    # Timestamp
    timestamp = datetime.now()

    return {
        "user_id": user_params['user_id'],
        "edition_id": edition_params['edition_id'],
        "general_rating": general_rating,
        "host_rating": host_rating,
        "difficulty_rating": difficulty_rating,
        "duration_rating": duration_rating,
        "number_of_people_rating": number_of_people_rating,
        "timestamp": timestamp
    }
