import { getDayOfYear } from "@/lib/getDayOfYear";
import { promises as fs } from "fs";
import { NextResponse } from "next/server";
import path from "path";

export async function GET(request: Request) {
  const url = new URL(request.url);
  const clientDate = url.searchParams.get("date");

  try {
    const filePath = path.join(process.cwd(), "public", "gibberish.json");
    const data = await fs.readFile(filePath, "utf-8");
    const phrases = JSON.parse(data);

    // Parse client date as UTC
    const clientDateObj = new Date(clientDate + "T00:00:00Z");
    const clientDayOfYear = getDayOfYear(clientDateObj);

    // Get server dayOfYear
    const serverDateObj = new Date();
    const serverDayOfYear = getDayOfYear(serverDateObj);

    // Check that the difference between server and client day is <= 1
    if (Math.abs(serverDayOfYear - clientDayOfYear) > 1) {
      return NextResponse.json(
        { error: "Client date is too far from server date" },
        { status: 400 }
      );
    }

    // Calculate the phrase index
    const todayIndex = clientDayOfYear % phrases.length;
    const todayPhrase = phrases[todayIndex];

    // Generate wordLengths string
    const wordLengths = todayPhrase.phrase
      .split(" ")
      .map((word: string) => "a".repeat(word.length))
      .join(" ");

    return NextResponse.json({
      phrase: todayPhrase.gibberish,
      wordLengths: wordLengths,
      hint: todayPhrase.hint,
    });
  } catch (error) {
    console.error("Error loading gibberish.json:", error);
    return NextResponse.json(
      { error: "Failed to load gibberish" },
      { status: 500 }
    );
  }
}
