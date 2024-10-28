"use client";

import { calculateTimeRemaining } from "@/lib/calculateTimeRemaining";
import { Flex, Heading, Text } from "@chakra-ui/react";
import { useRouter } from "next/navigation";
import { useEffect, useState } from "react";

export default function Completed() {
  const router = useRouter();
  const [timeRemaining, setTimeRemaining] = useState<string>("");

  useEffect(() => {
    // Update time remaining every second
    const updateRemainingTime = () =>
      setTimeRemaining(calculateTimeRemaining());
    updateRemainingTime();
    const timerInterval = setInterval(updateRemainingTime, 1000);

    // Check timestamp
    const completionDate = localStorage.getItem("gibberleCompletionDate");
    const today = new Date().toISOString().split("T")[0];

    if (completionDate !== today) {
      router.push("/");
    }

    // on unmount...
    return () => clearInterval(timerInterval);
  }, [router]);

  return (
    <>
      <Heading
        bgGradient="linear(to-r, #E18D6F, #D270BC)"
        bgClip="text"
        fontWeight="bold"
        fontSize="6vh"
        textAlign="center"
        position="relative"
        mt="2rem"
      >
        Gibberle
      </Heading>
      <Flex
        bg="brand.50"
        maxW="container.xl"
        mx="auto"
        mt={20}
        px={5}
        alignItems="center"
        justifyContent="center"
        flexDirection="column"
      >
        <Heading
          textAlign="center"
          fontSize={{ base: "3xl", sm: "3xl", md: "4xl", lg: "5xl", xl: "6xl" }}
          mb={5}
        >
          You have completed the daily Gibberle.
        </Heading>
        <Heading
          textAlign="center"
          fontSize={{ base: "xl", sm: "lg", md: "2xl", lg: "3xl", xl: "4xl" }}
          bgGradient="linear(to-r, #E18D6F, #D270BC)"
          bgClip="text"
          mb={5}
        >
          Come back at 12 AM local time to do the next one!
        </Heading>
        <Text fontSize={{ base: "md", sm: "md", md: "xl" }} color="gray.500">
          Next game available in: {timeRemaining}
        </Text>
      </Flex>
    </>
  );
}
