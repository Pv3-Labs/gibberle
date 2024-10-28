import { InputPhrase } from "@/components/InputPhrase/InputPhrase";
import { Box, Heading } from "@chakra-ui/react";

export default function Home() {
  return (
    <Box
      display="flex"
      flexDirection="column"
      alignItems="center" // Center horizontally
      minHeight="100vh" // Take full viewport height
    >
      <Heading
        size={{ base: "xl", md: "2xl", lg: "3xl", xl: "4xl" }}
        p={5}
        textAlign={"center"}
        mb={5}
        mt={{ base: "none", md: "30" }}
      >
        Gibberle
      </Heading>
      <Box marginY={5}>
        <InputPhrase correctPhrase="never gonna give you up" />
      </Box>
    </Box>
  );
}
