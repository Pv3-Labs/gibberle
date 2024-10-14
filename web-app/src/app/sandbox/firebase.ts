// sandbox/firebase.ts
import { initializeApp } from 'firebase/app';
import { getFirestore } from 'firebase/firestore';

const firebaseConfig = {
  apiKey: "AIzaSyBbgM3EjWKo_1FM_j5-X_Xk45Iy2-yNqEc",
  authDomain: "pv3-gibberle.firebaseapp.com",
  projectId: "pv3-gibberle",
  storageBucket: "pv3-gibberle.appspot.com",
  messagingSenderId: "873720026862",
  appId: "1:873720026862:web:6a4da4a5c96ca742f785ff",
  measurementId: "G-8MLEH3DDB4"
};


// Initialize Firebase
const app = initializeApp(firebaseConfig);
const db = getFirestore(app);

export { db };

