export function getDayOfYear() {
  const now = new Date();
  const start = new Date(now.getFullYear(), 0, 0); // Start of the year
  const diff = now.getTime() - start.getTime();
  const oneDay = 1000 * 60 * 60 * 24; // Milliseconds in a day
  return Math.floor(diff / oneDay);
}
