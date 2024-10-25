'use client'

import Navbar from "@/components/Navbar/Navbar";

export default function Home() {
  const handleHintClick = () => alert("Hint clicked!");
  const handleTutorialClick = () => alert("Tutorial clicked!");
  const handleStatsClick = () => alert("Stats clicked!");
  const handleSettingsClick = () => alert("Settings clicked!");

  return (
    <>
      <Navbar
          onHintClick={handleHintClick}
          onTutorialClick={handleTutorialClick}
          onStatsClick={handleStatsClick}
          onSettingsClick={handleSettingsClick}
          disabled={false}
        />
    </>
  );
}
