"use client";

import { db } from "@/lib/firebase/clientapp"; // Adjust path as necessary
import { Box, Button, Input, Stack, Text, useToast } from "@chakra-ui/react";
import { addDoc, collection } from "firebase/firestore";
import { useState } from "react";

interface InputField {
  name: string;
  value: string;
}

export function FirestoreDynamicInput() {
  const [inputs, setInputs] = useState<InputField[]>([
    { name: "name", value: "" },
  ]);
  const toast = useToast(); // Initialize toast hook

  const addInput = () => {
    setInputs([...inputs, { name: `Field ${inputs.length + 1}`, value: "" }]);
  };

  const handleInputChange = (
    index: number,
    event: React.ChangeEvent<HTMLInputElement>
  ) => {
    const newInputs = [...inputs];
    newInputs[index].value = event.target.value;
    setInputs(newInputs);
  };

  const handleSubmit = async () => {
    const data = inputs.reduce((acc, input) => {
      acc[input.name] = input.value;
      return acc;
    }, {} as { [key: string]: string });

    const loadingToastId = toast({
      title: "Submitting document...",
      description: "Please wait while your document is being submitted.",
      status: "info",
      duration: null, // Keep it open until success or error
      isClosable: true,
    });

    try {
      await addDoc(collection(db, "tests"), data);

      toast.update(loadingToastId, {
        title: "Document submitted",
        description: "Your document has been successfully submitted.",
        status: "success",
        duration: 3000,
        isClosable: true,
      });

      console.log("Document added:", data);
    } catch (e) {
      console.error("Error adding document: ", e);

      toast.update(loadingToastId, {
        title: "Submission failed",
        description: "There was an error submitting your document.",
        status: "error",
        duration: 3000,
        isClosable: true,
      });
    }
  };

  const generateRandomInputs = () => {
    const randomInputCount = Math.floor(Math.random() * 5) + 1; // Generate between 1-5 random inputs
    const randomInputs = Array.from({ length: randomInputCount }, (_, i) => ({
      name: `Field${i + 1}`,
      value: Math.random().toString(36).substring(7), // Random string
    }));

    const documentName = Math.random().toString(36).substring(1);

    setInputs([{ name: "name", value: documentName }, ...randomInputs]);
  };

  return (
    <Box p={5} shadow="md" borderWidth="1px" borderRadius="md" bg="#212121">
      <Text fontSize="xl" fontWeight="bold">
        Create new document
      </Text>
      <Stack spacing={3} mt={4}>
        {inputs.map((input, index) => (
          <Input
            key={index}
            placeholder={input.name}
            value={input.value}
            onChange={(event) => handleInputChange(index, event)}
          />
        ))}
        <Button onClick={addInput}>Add Input</Button>
        <Button onClick={generateRandomInputs}>Generate Random Inputs</Button>
        <Button colorScheme="blue" onClick={handleSubmit}>
          Submit to Firestore
        </Button>
      </Stack>
    </Box>
  );
}
