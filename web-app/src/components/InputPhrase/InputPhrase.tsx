"use client";  // Needed this line so useState, and useEffect work
import { InputCursor } from "@/components/InputCursor/InputCursor";
import { Box, Text } from "@chakra-ui/react";
import { useEffect, useState } from "react";
import { InputPhraseProp } from "./types";

export const InputPhrase: React.FC<InputPhraseProp> = ({correctPhrase}) => {
    // Note for this code: "\u00A0" is a non-breaking space, 
    //  if you don't use it adding spaces in certain scenarios
    //  won't do anything as it will collapses the spaces.
    //  So I needed to use this special space character 
    //  ONLY in place of the characters, not the regular spaces
    const [inputStr, setInputStr] = useState<string>(
        correctPhrase
            .split("")
            .map((char) => (char === " " ? " " : "\u00A0"))
            .join("")
    );
    const [inputIndex, setInputIndex] = useState<number>(0);
    const [cursorIsVisible, setCursorIsVisible] = useState<boolean>(true);
    const [caretStr, setCaretStr] = useState<string>("_" + 
        correctPhrase
            .slice(1)
            .split("")
            .map((char) => (char === " " ? " " : "\u00A0"))
            .join("")
    );

    const blankPhrase = correctPhrase
        .split("")
        .map((char) => (char === " " ? " " : "_"))
        .join("");
    
    const handleKeyDown = (event: KeyboardEvent) => {
        console.log("inputIndex: " + inputIndex);
        const key = event.key.toUpperCase();
        if ("QWERTYUIOPASDFGHJKLZXCVBNM".includes(key) && (inputIndex < correctPhrase.length)) {
            setInputStr((prev) => prev.slice(0, inputIndex) + key + prev.slice(inputIndex + 1));
            if (correctPhrase[inputIndex + 1] === " ") {
                // Next character after the cursor is a space, so add a space automatically
                setCaretStr((prev) => prev.slice(0, inputIndex) + "\u00A0 " + "_" + prev.slice(inputIndex + 3));
                setInputIndex((prev) => prev + 2);
            } else {
                if (inputIndex === correctPhrase.length - 1) {
                    setCaretStr((prev) => prev.slice(0, inputIndex) + "_");
                } else {
                    setCaretStr((prev) => prev.slice(0, inputIndex) + "\u00A0" + "_" + prev.slice(inputIndex + 2));
                }
                setInputIndex((prev) => prev + 1);
            }
        } else if (key === "BACKSPACE" && (inputIndex > 0)) {
            if (inputStr[inputIndex - 1] === " ") {
                // Previouse character is a space, so automatically delete space
                setInputStr((prev) => prev.slice(0, inputIndex-2) + "\u00A0 " + prev.slice(inputIndex));
                if (inputIndex === correctPhrase.length - 1) {
                    setCaretStr((prev) => prev.slice(0, inputIndex-2) + "_ " + "\u00A0 " + prev.slice(inputIndex + 1));
                } else {
                    setCaretStr((prev) => prev.slice(0, inputIndex-2) + "_ \u00A0" + prev.slice(inputIndex + 1));
                }
                setInputIndex((prev) => prev - 2);
            } else {
                setInputStr((prev) => prev.slice(0, inputIndex-1) + "\u00A0" + prev.slice(inputIndex));
                if (inputIndex === correctPhrase.length) {
                    setCaretStr((prev) => prev.slice(0, inputIndex-1) + "_");
                } else {
                    setCaretStr((prev) => prev.slice(0, inputIndex-1) + "_" + "\u00A0" + prev.slice(inputIndex + 1));
                }
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
        if (inputIndex === correctPhrase.length) {
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