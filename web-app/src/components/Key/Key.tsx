import { KeyProp } from "@/components/Key/types";
import { Button, Text, useBreakpointValue } from "@chakra-ui/react";

export const Key: React.FC<KeyProp & { onClick: () => void }> = ({
  keyChar,
  isPressed = false,
  onClick,
}) => {
  // Enter Key is different width than the rest of the keys
  // THOUGHT: should we do something where each key has a fixed padding
  //  instead of a width attribute? Or something like it?
  //  - this would make it so we don't have to set the width
  const displayChar = keyChar === "Enter" ? "Ent" : keyChar;
  const specialChar = keyChar === "Enter" || keyChar === "Del";

  const keyHeight = useBreakpointValue({
    base: "7vh",
    sm: "8vh",
    md: "9vh",
    lg: "10vh",
  });

  return (
    <Button
      bg="#292929"
      w={specialChar ? "5.5vw" : "4.5vw"}
      h={keyHeight}
      minW="32px" // Prevent keys from becoming too small
      minH="32px"
      borderRadius={12}
      // boxShadow is the gradient glow around the key when pressed
      boxShadow={
        isPressed
          ? "-2px -2px 3px 1px #E18D6F, 2px 2px 3px 1px #D270BC, -2px 2px 3px 1px #E18D6F, 2px -2px 3px 1px #D270BC"
          : "0 0 0 0px transparent"
      }
      _hover={{ bg: "#333333" }}
      display="flex"
      alignItems="center"
      justifyContent="center"
      onClick={onClick} // Added onClick handler for mobile and desktop interaction
    >
      <Text
        fontSize="min(2.5vw, 36px)"
        fontWeight="medium"
        bgGradient={isPressed ? "linear(to-r, #E18D6F, #D270BC)" : "undefined"}
        bgClip="text"
        color={isPressed ? "transparent" : "#FFFFFF"}
      >
        {displayChar}
      </Text>
    </Button>
  );
};
