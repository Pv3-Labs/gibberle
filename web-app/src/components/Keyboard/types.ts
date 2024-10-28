export interface KeyboardProp {
  layout?: string; // keyboard layout (qwerty, etc.)
  isHidden?: boolean; // whether or not to hide keyboard
  isDisabled?: boolean; // disable keyboard for things like popups
  onKeyPress: (key: string) => void;
}
