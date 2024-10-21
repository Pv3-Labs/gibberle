"use client";

import {
  onAuthStateChangedF,
  signInWithGoogle,
  signOut,
} from "@/lib/firebase/auth";
import { Box, Button, Text } from "@chakra-ui/react";
import { User } from "firebase/auth";
import { useEffect, useState } from "react";

export function FirebaseAuth() {
  const [user, setUser] = useState<User | null>(null);

  // Listen for auth state changes
  useEffect(() => {
    const unsubscribe = onAuthStateChangedF((authUser) => {
      if (authUser) {
        setUser(authUser as User);
      } else {
        setUser(null);
      }
    });

    return () => unsubscribe();
  }, []);

  // handle sign-in with Google
  const handleSignIn = async () => {
    try {
      await signInWithGoogle();
    } catch (error) {
      console.error("Error during sign-in", error);
    }
  };

  // handle sign-out
  const handleSignOut = async () => {
    try {
      await signOut();
    } catch (error) {
      console.error("Error during sign-out", error);
    }
  };

  return (
    <Box>
      {user ? (
        <Box>
          <Text fontSize="xl" mb={5}>
            Hello, {user.displayName}
          </Text>
          <Button colorScheme="red" onClick={handleSignOut}>
            Sign Out
          </Button>
        </Box>
      ) : (
        <Button colorScheme="blue" onClick={handleSignIn}>
          Sign in with Google
        </Button>
      )}
    </Box>
  );
}
