import {
  GoogleAuthProvider,
  onAuthStateChanged,
  signInWithPopup,
} from "firebase/auth";
import { auth } from "./clientapp";

// Function to listen to authentication state changes
export function onAuthStateChangedF(cb: (user: unknown) => void) {
  return onAuthStateChanged(auth, cb);
}

// Function to sign in with Google
export async function signInWithGoogle() {
  const provider = new GoogleAuthProvider();
  try {
    await signInWithPopup(auth, provider);
  } catch (error) {
    console.error("Error signing in with Google", error);
  }
}

// Function to sign out the current user
export async function signOut() {
  try {
    return await auth.signOut();
  } catch (error) {
    console.error("Error signing out", error);
  }
}
