"use client";

import GibberishPhrase from "@/components/GibberishPhrase/GibberishPhrase";
import { InputPhrase } from "@/components/InputPhrase/InputPhrase";
import { Keyboard } from "@/components/Keyboard/Keyboard";
import Navbar from "@/components/Navbar/Navbar";
import { Box, VStack } from "@chakra-ui/react";
import { useEffect, useState } from "react";

export default function Home() {
  const [gibberishData, setGibberishData] = useState<{
    phrase: string;
    wordLengths: string;
  } | null>(null);

  useEffect(() => {
    const fetchGibberishData = async () => {
      const response = await fetch("/api/get-gibberish");
      const data = await response.json();
      setGibberishData(data);
    };

    fetchGibberishData();
  }, []);

  if (!gibberishData) {
    return <p>Loading...</p>; // Loading state while data is fetched
  }

  return (
    <>
      <Navbar />
      <Box bg="brand.50" maxW="container.xl" mx="auto" px={5} overflow="hidden">
        <VStack align="center" w="full">
          <GibberishPhrase phrase={gibberishData.phrase} />
          <InputPhrase wordLengths={gibberishData.wordLengths} />{" "}
        </VStack>
      </Box>
      <Keyboard isHidden={false} isDisabled={false} />
    </>
  );
}
