// Import the functions you need from the SDKs you need
import { Analytics, getAnalytics } from "firebase/analytics";
import { initializeApp } from "firebase/app";
import { Auth, getAuth } from "firebase/auth";
import { Firestore, getFirestore } from "firebase/firestore";
// import { getMessaging, Messaging } from "firebase/messaging";
// import { FirebaseStorage, getStorage } from "firebase/storage";

// Your web app's Firebase configuration
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
const analytics: Analytics = getAnalytics(app);

// Export Firebase services
export const auth: Auth = getAuth(app);
export const db: Firestore = getFirestore(app);
export const firebaseApp = app; // Optionally export the app instance.
// export const storage: FirebaseStorage = getStorage(app);
// export const messaging: Messaging = getMessaging(app);