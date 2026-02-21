import express from "express";
import { getUserScore, recomputeUserScore } from "./score";

const app = express();

app.get("/users/:userId/score", (req, res) => {
  const score = getUserScore(req.params.userId);
  res.json({ userId: req.params.userId, score });
});

app.post("/users/:userId/score/recompute", (req, res) => {
  const score = recomputeUserScore(req.params.userId);
  res.json({ userId: req.params.userId, score });
});

export function createServer() {
  return app;
}
