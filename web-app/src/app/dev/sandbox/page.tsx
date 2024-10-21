"use client";

import { FirebaseAuth } from "@/components/dev/FirebaseAuth";
import { FirestoreDisplay } from "@/components/dev/FirestoreDisplay";
import { FirestoreDynamicInput } from "@/components/dev/FirestoreDynamicInput";
import { onAuthStateChangedF } from "@/lib/firebase/auth"; // Adjust path as needed
import { Box, Heading, Text } from "@chakra-ui/react";
import { useEffect, useState } from "react";

export default function Sandbox() {
  const [user, setUser] = useState<unknown>(null); // Track authenticated user

  useEffect(() => {
    const unsubscribe = onAuthStateChangedF((authUser) => {
      if (authUser) {
        setUser(authUser);
      } else {
        setUser(null);
      }
    });

    return () => unsubscribe();
  }, []);

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
      <FirebaseAuth />
      {user ? (
        <>
          <Box my={5}>
            <FirestoreDisplay />
          </Box>
          <FirestoreDynamicInput />
        </>
      ) : (
        <Text textAlign="center" mt={5}>
          Please sign in to access sandbox features.
        </Text>
      )}
    </Box>
  );
}
