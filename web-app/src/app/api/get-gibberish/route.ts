import { getDayOfYear } from "@/lib/getDayOfYear";
import { promises as fs } from "fs";
import { NextResponse } from "next/server";
import path from "path";

export async function GET() {
  try {
    const filePath = path.join(process.cwd(), "public", "gibberish.json");
    const data = await fs.readFile(filePath, "utf-8");
    const phrases = JSON.parse(data);

    const dayOfYear = getDayOfYear();
    const todayIndex = dayOfYear % phrases.length;

    const todayPhrase = phrases[todayIndex];

    // Generate wordLengths string
    const wordLengths = todayPhrase.phrase
      .split(" ")
      .map((word: string | unknown[]) => "a".repeat(word.length))
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
