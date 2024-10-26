"use client";  // Needed this line so useState, and useEffect work
import { InputCursor } from "@/components/InputCursor/InputCursor";
import { Box, Text } from "@chakra-ui/react";
import { useEffect, useState } from "react";
import { InputPhraseProp } from "./types";

export const InputPhrase: React.FC<InputPhraseProp> = ({correctPhrase}) => {
    // variables for tracking characters per word and which word the user is at

    const [inputStr, setInputStr] = useState<string>("");
    const [caretStr, setCaretStr] = useState<string>("_");

    const blankPhrase = correctPhrase
        .split("")
        .map((char) => (char === " " ? " " : "_"))
        .join("");
    
    const handleKeyDown = (event: KeyboardEvent) => {
        const key = event.key.toLowerCase();
        if ("qwertyuiopasdfghjklzxcvbnm".includes(key)) {
            setInputStr((prev) => prev + key);
            // "\u00A0" is a non-breaking space, if you just do " ",
            // it collapses the spaces so nothing happens to the caretStr variable
            setCaretStr((prev) => "\u00A0" + prev);
        } else if (key === "backspace") {
            setInputStr((prev) => prev.slice(0, -1));
            setCaretStr((prev) => prev.slice(1));
        }
    };

    useEffect(() => {
        window.addEventListener("keydown", handleKeyDown);
        return () => {
            window.removeEventListener("keydown", handleKeyDown);
        }
    }, []);

    return (
        <Box position='relative' display='inline-block'>
            <Text
                fontFamily='Courier New'
                fontSize={40}
                letterSpacing='0.75rem'
                position='absolute'
                top='0.3rem' // Adjust this value to control the height offset
                left={0}
                color='#EAEAEA' // Adjust to desired text color
                textAlign='left'
                boxShadow='0 0 2px red'
            >
                {inputStr}
            </Text>
            <Text
                fontFamily='Courier New'
                fontSize={40}
                textAlign='left'
                letterSpacing='0.75rem'
            >
                {blankPhrase}
            </Text>
            <Box position='absolute' top={0} left={0}>
                <InputCursor cursorStr={caretStr} />
            </Box>
        </Box>
    );
}   