export function calculateTimeRemaining(): string {
  const now = new Date();
  const nextMidnight = new Date(
    now.getFullYear(),
    now.getMonth(),
    now.getDate() + 1,
    0,
    0,
    0 // 12 AM of the next day
  );
  const timeUntilMidnight = nextMidnight.getTime() - now.getTime();

  // Convert time to hours, minutes, and seconds
  const hours = Math.floor(timeUntilMidnight / (1000 * 60 * 60));
  const minutes = Math.floor(
    (timeUntilMidnight % (1000 * 60 * 60)) / (1000 * 60)
  );
  const seconds = Math.floor((timeUntilMidnight % (1000 * 60)) / 1000);

  return `${hours.toString().padStart(2, "0")}:${minutes
    .toString()
    .padStart(2, "0")}:${seconds.toString().padStart(2, "0")}`;
}
