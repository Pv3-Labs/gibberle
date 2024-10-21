"use client";

import { db } from "@/lib/firebase/clientapp"; // Adjust path as necessary
import { Box, Button, Input, Stack, Text } from "@chakra-ui/react";
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

    try {
      await addDoc(collection(db, "tests"), data);
      console.log("Document added:", data);
    } catch (e) {
      console.error("Error adding document: ", e);
    }
  };

  const generateRandomInputs = () => {
    const randomInputCount = Math.floor(Math.random() * 5) + 1; // Generate between 1-5 random inputs
    const randomInputs = Array.from({ length: randomInputCount }, (_, i) => ({
      name: `RandomField${i + 1}`,
      value: Math.random().toString(36).substring(7), // Random string
    }));

    setInputs([{ name: "name", value: "Random Name" }, ...randomInputs]);
  };

  return (
    <Box p={5} shadow="md" borderWidth="1px" borderRadius="md">
      <Text fontSize="xl" fontWeight="bold">
        Dynamic Input Card
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
