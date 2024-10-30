import { getDayOfYear } from "@/lib/getDayOfYear";
import { promises as fs } from "fs";
import { NextResponse } from "next/server";
import path from "path";

export async function POST(request: Request) {
  try {
    const { guess, date } = await request.json();

    if (!date) {
      return NextResponse.json(
        { error: "Date parameter is required" },
        { status: 400 }
      );
    }

    // Get server's current date as a Date object
    const serverDateObj = new Date();
    const clientDateObj = new Date(date + "T00:00:00Z"); // Parse client date as UTC midnight

    // Calculate the difference in days between server and client dates
    const differenceInDays = Math.abs(
      Math.floor(
        (serverDateObj.getTime() - clientDateObj.getTime()) /
          (1000 * 60 * 60 * 24)
      )
    );

    // Reject if the difference is more than one day
    if (differenceInDays > 1) {
      return NextResponse.json(
        { error: "Date difference is too large" },
        { status: 403 }
      );
    }

    // Proceed to validate the guess for the client
    const filePath = path.join(process.cwd(), "public", "gibberish.json");
    const data = await fs.readFile(filePath, "utf-8");
    const phrases = JSON.parse(data);

    // Calculate day index based on client date
    const clientDayOfYear = getDayOfYear(clientDateObj);
    const indexForClientDate = clientDayOfYear % phrases.length;

    const correctPhrase = phrases[indexForClientDate].phrase;

    // Check if the guess matches the correct phrase
    const isCorrect = guess.toLowerCase() === correctPhrase.toLowerCase();
    return NextResponse.json({ correct: isCorrect });
  } catch (error) {
    console.error("Error validating guess:", error);
    return NextResponse.json(
      { error: "Failed to validate guess" },
      { status: 500 }
    );
  }
}
