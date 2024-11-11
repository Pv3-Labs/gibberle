export interface KeyProp {
  keyChar: string; // The letter/word the key is
  isPressed?: boolean; // true when a key is pressed
  onClick: () => void;
}
