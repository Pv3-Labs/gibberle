// sandbox/testFirestore.ts
import { addDoc, collection } from 'firebase/firestore'; // Import Firestore methods
import { db } from './firebase'; // Import the Firestore instance

export const testFirestore = async () => {
  try {
    // Define the data you want to add to the 'test' collection
    const testData = {
      name: 'Test Document',
      createdAt: new Date(),
    };

    // Reference the 'test' collection
    const testCollectionRef = collection(db, 'test');

    // Add a new document with the data
    const docRef = await addDoc(testCollectionRef, testData);
    console.log('Document written with ID: ', docRef.id);
  } catch (error) {
    console.error('Firestore error:', error);
  }
};
