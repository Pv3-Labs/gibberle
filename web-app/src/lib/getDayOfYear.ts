export function getDayOfYear(date: Date): number {
  const start = new Date(Date.UTC(date.getUTCFullYear(), 0, 0)); // Start of the year in UTC
  const diff = date.getTime() - start.getTime();
  const oneDay = 1000 * 60 * 60 * 24; // Milliseconds in a day
  return Math.floor(diff / oneDay);
}
