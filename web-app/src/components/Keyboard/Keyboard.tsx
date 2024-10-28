"use client"; // Needed this line so useRef, useState, and useEffect worked
import { Key } from "@/components/Key/Key";
import { KeyboardProp } from "@/components/Keyboard/types";
import { Box, HStack, useBreakpointValue, VStack } from "@chakra-ui/react";
import { useEffect, useRef, useState } from "react";

export const Keyboard: React.FC<KeyboardProp> = ({
  layout = "qwerty",
  isHidden = false,
  isDisabled = false,
}) => {
  // Default keyboard layout
  const qwertyKeyboard = [
    ["Q", "W", "E", "R", "T", "Y", "U", "I", "O", "P"],
    ["A", "S", "D", "F", "G", "H", "J", "K", "L"],
    ["Enter", "Z", "X", "C", "V", "B", "N", "M", "Del"]
  ];

  // set of pressed keys
  const [pressedKeys, setPressedKeys] = useState<Set<string>>(new Set());
  // hold the latest values of pressedKeys without causing a re-render
  const pressedKeysRef = useRef(pressedKeys);

  // function that updates the list of pressed keys,
  // also updates the ref to make sure it matches
  const updatePressedKeys = (keys: Set<string>) => {
    setPressedKeys(new Set(keys));
    pressedKeysRef.current = keys;
  };

  // key press event handler
  const handleKeyDown = (event: KeyboardEvent) => {
    // Don't want 'Enter' to be all uppercase to make passing it as a prop attribute easier
    let key = event.key === "Enter" ? "Enter" : event.key.toUpperCase();
    if (key === "BACKSPACE") {
      key = "Del";
    }
    if ("QWERTYUIOPASDFGHJKLZXCVBNM".includes(key) || key === "Enter" || key === "Del") {
      const newKeys = new Set(pressedKeysRef.current);
      // Error handeling?
      newKeys.add(key);
      updatePressedKeys(newKeys);
    }
  };

  // event handler for when a key is released from being pressed
  const handleKeyUp = (event: KeyboardEvent) => {
    // Don't capitalize 'Enter' because we don't capitalize it on KeyDown handler
    let key = event.key === "Enter" ? "Enter" : event.key.toUpperCase();
    if (key === "BACKSPACE") {
      key = "Del";
    }
    if ("QWERTYUIOPASDFGHJKLZXCVBNM".includes(key) || key === "Enter" || key === "Del") {
      const newKeys = new Set(pressedKeysRef.current);
      // Error handeling??
      newKeys.delete(key);
      updatePressedKeys(newKeys);
    }
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
        borderRadius={"36px"}
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
                />
              ))}
            </HStack>
          ))}
        </VStack>
      </Box>
    </Box>
  );
};
