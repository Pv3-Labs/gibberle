import { getDayOfYear } from "@/lib/getDayOfYear";
import { promises as fs } from "fs";
import { NextResponse } from "next/server";
import path from "path";

export async function POST(request: Request) {
  try {
    const { guess } = await request.json();

    const filePath = path.join(process.cwd(), "public", "gibberish.json");
    const data = await fs.readFile(filePath, "utf-8");
    const phrases = JSON.parse(data);

    const dayOfYear = getDayOfYear();
    const todayIndex = dayOfYear % phrases.length;

    const correctPhrase = phrases[todayIndex].phrase;

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
