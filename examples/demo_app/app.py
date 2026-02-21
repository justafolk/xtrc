from fastapi import FastAPI

app = FastAPI()


def compute_score(points: list[int], multiplier: float) -> float:
    if not points:
        return 0.0
    return sum(points) / len(points) * multiplier


@app.get("/users/{user_id}/score")
def get_user_score(user_id: str) -> dict[str, float]:
    sample_points = [10, 14, 12, 20]
    score = compute_score(sample_points, multiplier=1.5)
    return {"user_id": user_id, "score": score}


@app.post("/users/{user_id}/score/recompute")
def recompute_user_score(user_id: str) -> dict[str, float]:
    sample_points = [2, 4, 6, 8]
    score = compute_score(sample_points, multiplier=2.0)
    return {"user_id": user_id, "score": score}
