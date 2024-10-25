import { InputPhrase } from "@/components/InputPhrase/InputPhrase";
import { Box, Heading } from "@chakra-ui/react";

export default function Home() {
  return (
    <Box bg="brand.50" maxW="container.xl" mx="auto" px={5}>
      <Heading
        size={{ base: "xl", md: "2xl", lg: "3xl", xl: "4xl" }}
        p={5}
        textAlign={"center"}
        mb={5}
        mt={{ base: "none", md: "30" }}
      >
        Gibberle
      </Heading>
      <InputPhrase correctPhrase="never gonna give you up" />
    </Box>
  );
}
