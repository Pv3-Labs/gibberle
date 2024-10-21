"use client";  // Needed this line so useRef, useState, and useEffect worked
import { Key } from "@/components/Key/Key";
import { KeyboardProp } from "@/components/Keyboard/types";
import { Box, HStack, VStack } from "@chakra-ui/react";
import { useEffect, useRef, useState } from "react";

export const Keyboard: React.FC<KeyboardProp> = ({layout="qwerty"}) => {
  // Default keyboard layout
  const qwertyKeyboard = [["Q", "W", "E", "R", "T", "Y", "U", "I", "O", "P"],
                          ["A", "S", "D", "F", "G", "H", "J", "K", "L"],
                          ["Z", "X", "C", "V", "B", "N", "M", "Enter"]];

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
    const key = (event.key === "Enter") ? "Enter" : event.key.toUpperCase();
    if ("QWERTYUIOPASDFGHJKLZXCVBNM".includes(key) || key === "Enter") {
      const newKeys = new Set(pressedKeysRef.current);
      // Error handeling?
      newKeys.add(key);
      updatePressedKeys(newKeys);
    }
  };

  // event handler for when a key is released from being pressed
  const handleKeyUp = (event: KeyboardEvent) => {
    // Don't capitalize 'Enter' because we don't capitalize it on KeyDown handler
    const key = (event.key === "Enter") ? "Enter" : event.key.toUpperCase();
    if ("QWERTYUIOPASDFGHJKLZXCVBNM".includes(key) || key === "Enter") {
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

  return (
    <Box
      w='850px'
      h='360px'
      borderRadius={'36px'}
      background={`linear-gradient(rgba(0, 0, 0, 0.33), rgba(0, 0, 0, 0.33)), #A199CA`}
      backgroundBlendMode="multiply"
      display="flex"
      justifyContent="center"
      alignItems="center"
    >
      <VStack align={'center'}>
        {qwertyKeyboard.map((row, rowIndex) => (
          <HStack key={rowIndex}>
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
  );
};
