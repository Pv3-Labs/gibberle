"use client";

import { db } from "@/lib/firebase/clientapp";
import { Box, Button, SimpleGrid, Spinner, Text } from "@chakra-ui/react";
import { collection, getDocs } from "firebase/firestore";
import { useEffect, useState } from "react";

interface FirestoreDocument {
  id: string;
  [key: string]: any;
}

export function FirestoreDisplay() {
  const [documents, setDocuments] = useState<FirestoreDocument[]>([]);
  const [loading, setLoading] = useState<boolean>(true);

  const fetchDocuments = async (): Promise<void> => {
    setLoading(true);
    try {
      const querySnapshot = await getDocs(collection(db, "tests"));
      const docsArray = querySnapshot.docs.map((doc) => ({
        id: doc.id,
        ...doc.data(),
      })) as FirestoreDocument[];
      setDocuments(docsArray);
      setLoading(false);
    } catch (e) {
      console.error("Error fetching documents: ", e);
      setLoading(false);
    }
  };

  useEffect(() => {
    fetchDocuments(); // Initial fetch on page load
  }, []);

  if (loading) {
    return <Spinner />;
  }

  return (
    <>
      <Button colorScheme="green" onClick={fetchDocuments} mb={4}>
        Refresh Documents
      </Button>
      <SimpleGrid columns={{ sm: 1, md: 2, lg: 3 }} spacing={6}>
        {documents.map((doc) => (
          <Box
            key={doc.id}
            p={5}
            shadow="md"
            borderWidth="1px"
            borderRadius="md"
            bg="#212121"
          >
            <Text fontSize="xl" fontWeight="bold">
              {doc.name}
            </Text>
            <Text mb={2}>ID: {doc.id}</Text>
            <Box>
              {Object.keys(doc).map((key) =>
                key !== "id" && key !== "name" ? (
                  <Text key={key}>
                    {key}: {JSON.stringify(doc[key])}
                  </Text>
                ) : null
              )}
            </Box>
          </Box>
        ))}
      </SimpleGrid>
    </>
  );
}
