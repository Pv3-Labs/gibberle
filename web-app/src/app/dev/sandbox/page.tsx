import { Box, Heading } from "@chakra-ui/react";

export default function Sandbox() {
  return (
    <Box bg="brand.50" maxW="container.xl" mx="auto" px={5}>
      <Heading
        size={{ base: "xl", md: "2xl", lg: "3xl", xl: "4xl" }}
        p={5}
        textAlign={"center"}
        mb={5}
        mt={{ base: "none", md: "30" }}
      >
        Sandbox
      </Heading>
    </Box>
  );
}
