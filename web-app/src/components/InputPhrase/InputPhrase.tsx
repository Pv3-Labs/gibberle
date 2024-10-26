"use client";  // Needed this line so useState, and useEffect work
import { InputCursor } from "@/components/InputCursor/InputCursor";
import { Box, Text } from "@chakra-ui/react";
import { useEffect, useState } from "react";
import { InputPhraseProp } from "./types";

export const InputPhrase: React.FC<InputPhraseProp> = ({correctPhrase}) => {
    // variables for tracking characters per word and which word the user is at

    const [inputStr, setInputStr] = useState<string>("");
    const [caretStr, setCaretStr] = useState<string>("_");
    const [inputIndex, setInputIndex] = useState<number>(0);

    const blankPhrase = correctPhrase
        .split("")
        .map((char) => (char === " " ? " " : "_"))
        .join("");
    
    const handleKeyDown = (event: KeyboardEvent) => {
        // NOTE: "\u00A0" is a non-breaking space, if you just do " ",
        // it collapses the spaces so need to use it 
        //  when adding spaces to inputStr and caretStr
        const key = event.key.toLowerCase();
        if ("qwertyuiopasdfghjklzxcvbnm".includes(key) && (inputStr.length < correctPhrase.length)) {
            if (correctPhrase[inputIndex + 1] === " ") {
                // Next character after the cursor is a space, so add a space automatically
                setInputStr((prev) => prev + key + "\u00A0");
                setCaretStr((prev) => "\u00A0" + "\u00A0" + prev);
                setInputIndex((prev) => prev + 2);
            } else {
                setInputStr((prev) => prev + key);
                setCaretStr((prev) => "\u00A0" + prev);
                setInputIndex((prev) => prev + 1);
            }
        } else if (key === "backspace" && (inputStr.length > 0)) {
            if (inputStr[inputIndex - 1] === "\u00A0") {
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

    useEffect(() => {
        window.addEventListener("keydown", handleKeyDown);
        return () => {
            window.removeEventListener("keydown", handleKeyDown);
        }
    }, [inputIndex, inputStr]);

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