"use client";  // Needed this line so useState, and useEffect work
import { InputCursor } from "@/components/InputCursor/InputCursor";
import { Box, Text } from "@chakra-ui/react";
import { useEffect, useState } from "react";
import { InputPhraseProp } from "./types";

export const InputPhrase: React.FC<InputPhraseProp> = ({correctPhrase}) => {
    const [inputStr, setInputStr] = useState<string>("");
    const [inputIndex, setInputIndex] = useState<number>(0);
    const [cursorIsVisible, setCursorIsVisible] = useState<boolean>(true);
    const [caretStr, setCaretStr] = useState<string>("_");

    const blankPhrase = correctPhrase
        .split("")
        .map((char) => (char === " " ? " " : "_"))
        .join("");
    
    const handleKeyDown = (event: KeyboardEvent) => {
        // NOTE: "\u00A0" is a non-breaking space, if you just do " ",
        // it collapses the spaces so need to use it 
        //  when adding spaces to inputStr and caretStr
        const key = event.key.toUpperCase();
        if ("QWERTYUIOPASDFGHJKLZXCVBNM".includes(key) && (inputStr.length < correctPhrase.length)) {
            if (correctPhrase[inputIndex + 1] === " ") {
                // Next character after the cursor is a space, so add a space automatically
                setInputStr((prev) => prev + key + " ");
                setCaretStr((prev) => "\u00A0" + "\u00A0" + prev);
                setInputIndex((prev) => prev + 2);
            } else {
                setInputStr((prev) => prev + key);
                setCaretStr((prev) => "\u00A0" + prev);
                setInputIndex((prev) => prev + 1);
            }
        } else if (key === "BACKSPACE" && (inputStr.length > 0)) {
            if (inputStr[inputIndex - 1] === " ") {
                // previouse character is a space, so automatically delete space
                setInputStr((prev) => prev.slice(0, -2));
                setCaretStr((prev) => prev.slice(2));
                setInputIndex((prev) => prev - 2);
            } else {
                setInputStr((prev) => prev.slice(0, -1));
                setCaretStr((prev) => prev.slice(1));
                setInputIndex((prev) => prev - 1);
            }
        }
    };

    // Listener for keyboard input
    useEffect(() => {
        window.addEventListener("keydown", handleKeyDown);
        return () => {
            window.removeEventListener("keydown", handleKeyDown);
        }
    }, [inputIndex, inputStr, correctPhrase]);

    // Listiner for checking the input string length
    // Makes cursor/caret invisible if input string is fully filled out
    useEffect(() => {
        if (inputStr.length === correctPhrase.length) {
            setCursorIsVisible(false);
        } else {
            setCursorIsVisible(true);
        }
    }, [inputStr, correctPhrase]);

    return (
        <Box position='relative' display='inline-block'>
            <Text
                fontFamily="'Hack', monospace"
                fontSize={{ base: '20px', md: '2.5vw', lg: '3vw' }}
                letterSpacing='0.75rem'
                position='absolute'
                top={'-4px'}
                left={0}
                color='#EAEAEA'
                textAlign='center'
            >
                {inputStr}
            </Text>
            <Text
                fontFamily="'Hack', monospace"
                fontSize={{ base: '20px', md: '2.5vw', lg: '3vw' }}
                textAlign='center'
                letterSpacing='0.75rem'
                color='#EAEAEA'
            >
                {blankPhrase}
            </Text>
            <Box position='absolute' top={0} left={0}>
                <InputCursor cursorStr={caretStr} isVisible={cursorIsVisible} />
            </Box>
        </Box>
    );
}   