"use client";

import { Text } from "@chakra-ui/react";
import { GibberishPhraseProp } from "./types";

export default function GibberishPhrase({ phrase }: GibberishPhraseProp) {
  return (
    <Text
      fontSize={{ base: "3xl", sm: "3xl", md: "4xl", lg: "5xl", xl: "6xl" }}
      my={20}
      w="100%"
      textAlign="center"
      color="#f4f4f4"
    >
      {phrase}
    </Text>
  );
}
