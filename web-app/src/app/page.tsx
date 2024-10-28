"use client";

import GibberishPhrase from "@/components/GibberishPhrase/GibberishPhrase";
import { InputPhrase } from "@/components/InputPhrase/InputPhrase";
import { Keyboard } from "@/components/Keyboard/Keyboard";
import Navbar from "@/components/Navbar/Navbar";
import { Box, Spinner, Text, VStack } from "@chakra-ui/react";
import { useRouter } from "next/navigation";
import { useEffect, useState } from "react";

export default function Home() {
  const router = useRouter();
  const [gibberishData, setGibberishData] = useState<{
    phrase: string;
    wordLengths: string;
    hint: string;
  } | null>(null);
  const [showHint, setShowHint] = useState(false);

  useEffect(() => {
    // Check if the user has already completed today's game
    const completionDate = localStorage.getItem("gibberleCompletionDate");
    const today = new Date().toISOString().split("T")[0];

    if (completionDate === today) {
      router.push("/completed"); // Redirect if game already completed
      return;
    }

    const fetchGibberishData = async () => {
      const response = await fetch("/api/get-gibberish");
      const data = await response.json();
      setGibberishData(data);
    };

    fetchGibberishData();
  }, [router]);

  const handleHintClick = () => setShowHint((prev) => !prev);

  return (
    <>
      <Navbar onHintClick={handleHintClick} />
      <Box bg="brand.50" maxW="container.xl" mx="auto" px={5} overflow="hidden">
        {!gibberishData ? (
          <VStack align="center" w="full">
            <Spinner my={20} />
          </VStack>
        ) : (
          <VStack align="center" w="full">
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
            <InputPhrase wordLengths={gibberishData.wordLengths} />
          </VStack>
        )}
      </Box>
      <Keyboard isHidden={false} isDisabled={false} />
    </>
  );
}
