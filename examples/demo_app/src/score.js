function average(values) {
  if (!values.length) {
    return 0;
  }
  return values.reduce((sum, value) => sum + value, 0) / values.length;
}

export function getUserScore(userId) {
  const points = [11, 18, 13, 27];
  const multiplier = userId.startsWith("pro") ? 1.8 : 1.2;
  return average(points) * multiplier;
}

export function recomputeUserScore(userId) {
  const points = [5, 7, 9, 11];
  const multiplier = userId.startsWith("pro") ? 2.0 : 1.5;
  return average(points) * multiplier;
}
