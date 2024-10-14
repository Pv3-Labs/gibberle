// Import necessary Firestore functions
import { addDoc, collection } from "firebase/firestore";
import { db } from './firebase'; // Adjust the import based on your structure

// Function to add a document to the 'testing' collection
async function addTestDocument() {
    try {
        // Reference to the 'testing' collection
        const testCollectionRef = collection(db, "test");

        // Document data you want to add
        const newDocData = {
            name: "Test Document",
            description: "This is a test document.",
            createdAt: new Date(),
        };

        // Add document to the collection
        const docRef = await addDoc(testCollectionRef, newDocData);
        console.log("Document added with ID: ", docRef.id);
    } catch (e) {
        console.error("Error adding document: ", e);
    }
}

// Call the function to add the document
addTestDocument();
