"use client";

import { db } from "@/lib/firebase/clientapp";
import { Button } from "@chakra-ui/react";
import { addDoc, collection } from "firebase/firestore";

export function FirestoreButton() {
  const addDocument = async (): Promise<void> => {
    try {
      await addDoc(collection(db, "tests"), {
        name: "Aitan Obese",
        createdAt: new Date(),
      });
      console.log("Document successfully added!");
    } catch (e) {
      console.error("Error adding document: ", e);
    }
  };

  return <Button onClick={addDocument}>Add Document</Button>;
}
