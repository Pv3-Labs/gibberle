"use client";

import { Heading } from "@chakra-ui/react";
import { GibberishPhraseProp } from "./types";

export default function GibberishPhrase({ phrase }: GibberishPhraseProp) {
  return (
    <Heading size="2xl" my={20} w="100%" textAlign="center" color="#f4f4f4">
      {phrase}
    </Heading>
  );
}
