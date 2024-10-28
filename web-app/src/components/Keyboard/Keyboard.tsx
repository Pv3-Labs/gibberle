"use client";

import { Key } from "@/components/Key/Key";
import { KeyboardProp } from "@/components/Keyboard/types";
import { Box, HStack, useBreakpointValue, VStack } from "@chakra-ui/react";
import { useEffect, useRef, useState } from "react";

export const Keyboard: React.FC<KeyboardProp> = ({
  layout = "qwerty",
  isHidden = false,
  isDisabled = false,
  onKeyPress,
}) => {
  // These are just to avoid the eslint errors for deployment
  // Remove later
  console.log(layout);
  console.log(isDisabled);
  console.log(isHidden);

  // Default keyboard layout
  const qwertyKeyboard = [
    ["Q", "W", "E", "R", "T", "Y", "U", "I", "O", "P"],
    ["A", "S", "D", "F", "G", "H", "J", "K", "L"],
    ["Enter", "Z", "X", "C", "V", "B", "N", "M", "Del"],
  ];

  // Set of pressed keys
  const [pressedKeys, setPressedKeys] = useState<Set<string>>(new Set());
  // Hold the latest values of pressedKeys without causing a re-render
  const pressedKeysRef = useRef(pressedKeys);

  // Function that updates the list of pressed keys,
  // also updates the ref to make sure it matches
  const updatePressedKeys = (keys: Set<string>) => {
    setPressedKeys(new Set(keys));
    pressedKeysRef.current = keys;
  };

  // Key press event handler for physical keyboards
  const handleKeyDown = (event: KeyboardEvent) => {
    // Don't want 'Enter' to be all uppercase to make passing it as a prop attribute easier
    let key = event.key === "Enter" ? "Enter" : event.key.toUpperCase();
    if (key === "BACKSPACE") {
      key = "Del";
    }
    if (
      "QWERTYUIOPASDFGHJKLZXCVBNM".includes(key) ||
      key === "Enter" ||
      key === "Del"
    ) {
      const newKeys = new Set(pressedKeysRef.current);
      // Error handling?
      newKeys.add(key);
      updatePressedKeys(newKeys);
    }
  };

  // Event handler for when a key is released from being pressed
  const handleKeyUp = (event: KeyboardEvent) => {
    // Don't capitalize 'Enter' because we don't capitalize it on KeyDown handler
    let key = event.key === "Enter" ? "Enter" : event.key.toUpperCase();
    if (key === "BACKSPACE") {
      key = "Del";
    }
    if (
      "QWERTYUIOPASDFGHJKLZXCVBNM".includes(key) ||
      key === "Enter" ||
      key === "Del"
    ) {
      const newKeys = new Set(pressedKeysRef.current);
      // Error handling??
      newKeys.delete(key);
      updatePressedKeys(newKeys);
    }
  };

  // Map to store the timeout IDs for each key
  const keyTimeouts = useRef<Map<string, NodeJS.Timeout>>(new Map());

  // Touch/click event handler for virtual keys (to work on mobile)
  const handleVirtualKeyPress = (key: string) => {
    if (onKeyPress) {
      onKeyPress(key); // Call onKeyPress prop
    }

    const newKeys = new Set(pressedKeysRef.current);
    newKeys.add(key);
    updatePressedKeys(newKeys);

    // Clear any existing timeout for this key before setting a new one
    if (keyTimeouts.current.has(key)) {
      clearTimeout(keyTimeouts.current.get(key));
    }

    // Set a new timeout for this key and store it in the map
    const timeoutId = setTimeout(() => {
      const updatedKeys = new Set(pressedKeysRef.current);
      updatedKeys.delete(key);
      updatePressedKeys(updatedKeys);
      keyTimeouts.current.delete(key); // Remove timeout from map after it's executed
    }, 50);

    keyTimeouts.current.set(key, timeoutId); // Store the new timeout for this key
  };

  useEffect(() => {
    window.addEventListener("keydown", handleKeyDown);
    window.addEventListener("keyup", handleKeyUp);

    return () => {
      window.removeEventListener("keydown", handleKeyDown);
      window.removeEventListener("keyup", handleKeyUp);
    };
  }, []);

  // Responsive container sizes
  const containerWidth = useBreakpointValue({
    base: "98vw",
    sm: "90vw",
    md: "70vw",
    lg: "60vw",
  });

  const containerHeight = useBreakpointValue({
    base: "29vh",
    sm: "32vh",
    md: "35vh",
    lg: "39vh",
  });

  const keySpacing = useBreakpointValue({
    base: "0.3vh",
    sm: "0.6vh",
    md: "1vh",
  });

  const rowSpacing = useBreakpointValue({
    base: "0.3vh",
    sm: "0.6vh",
    md: "1vh",
  });

  const bottomSpacing = useBreakpointValue({
    base: "0",
    sm: "1.5vh",
    md: "2vh",
    lg: "2vh",
  });

  return (
    <Box
      position="fixed"
      bottom={bottomSpacing}
      left="50%"
      transform="translateX(-50%)"
      width="100%"
      display="flex"
      justifyContent="center"
    >
      <Box
        w={containerWidth}
        h={containerHeight}
        borderRadius="2xl"
        // background={`linear-gradient(rgba(0, 0, 0, 0.33), rgba(0, 0, 0, 0.33)), #A199CA`}
        background={`transparent`}
        backgroundBlendMode="multiply"
        display="flex"
        justifyContent="center"
        alignItems="center"
      >
        <VStack spacing={rowSpacing}>
          {qwertyKeyboard.map((row, rowIndex) => (
            <HStack key={rowIndex} spacing={keySpacing}>
              {row.map((keyChar) => (
                <Key
                  key={keyChar}
                  keyChar={keyChar}
                  isPressed={pressedKeys.has(keyChar)}
                  onClick={() => handleVirtualKeyPress(keyChar)} // Add onClick handler
                />
              ))}
            </HStack>
          ))}
        </VStack>
      </Box>
    </Box>
  );
};
