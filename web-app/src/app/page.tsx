"use client";

import GibberishPhrase from "@/components/GibberishPhrase/GibberishPhrase";
import { InputPhrase } from "@/components/InputPhrase/InputPhrase";
import { Keyboard } from "@/components/Keyboard/Keyboard";
import Navbar from "@/components/Navbar/Navbar";
import { Box, Spinner, Text, VStack } from "@chakra-ui/react";
import { useRouter } from "next/navigation";
import { useEffect, useRef, useState } from "react";

export default function Home() {
  const router = useRouter();
  const [gibberishData, setGibberishData] = useState<{
    phrase: string;
    wordLengths: string;
    hint: string;
  } | null>(null);
  const [showHint, setShowHint] = useState(false);

  const inputPhraseRef = useRef<{ processKey: (key: string) => void } | null>(
    null
  );

  const handleKeyPress = (key: string) => {
    if (inputPhraseRef.current) {
      inputPhraseRef.current.processKey(key);
    }
  };

  useEffect(() => {
    // Check if the user has already completed today's game
    const completionDate = localStorage.getItem("gibberleCompletionDate");
    const today = new Date().toLocaleDateString("en-CA");

    console.log(today);

    if (completionDate === today) {
      router.push("/completed"); // Redirect if game already completed
      return;
    }

    const fetchGibberishData = async () => {
      const response = await fetch(`/api/get-gibberish?date=${today}`);
      const data = await response.json();
      setGibberishData(data);
    };

    fetchGibberishData();
  }, [router]);

  const handleHintClick = () => setShowHint((prev) => !prev);

  return (
    <Box height="100vh">
      <Navbar
        onHintClick={handleHintClick}
        onTutorialClick={() =>
          console.log("This feature has not been implemented yet!")
        }
        onStatsClick={() =>
          console.log("This feature has not been implemented yet!")
        }
        onSettingsClick={() =>
          console.log("This feature has not been implemented yet!")
        }
      />
      <Box bg="brand.50" maxW="container.xl" mx="auto" px={5} overflow="hidden">
        {!gibberishData ? (
          <VStack align="center" w="full" mb={10}>
            <Spinner my={20} />
          </VStack>
        ) : (
          <VStack align="center" w="full" mb={10}>
            <GibberishPhrase phrase={gibberishData.phrase} />
            {showHint && (
              <Text
                fontSize={{ base: "md", md: "lg", lg: "xl" }}
                textAlign="center"
                color="#b9b9b9"
                mb={3}
              >
                Hint: {gibberishData.hint}
              </Text>
            )}
            <InputPhrase
              ref={inputPhraseRef}
              wordLengths={gibberishData.wordLengths}
              handleKeyPress={handleKeyPress}
            />
          </VStack>
        )}
      </Box>
      <Keyboard
        isHidden={false}
        isDisabled={false}
        onKeyPress={handleKeyPress}
      />
    </Box>
  );
}
