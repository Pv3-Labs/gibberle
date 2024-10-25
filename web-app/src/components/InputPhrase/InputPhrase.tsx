import { InputCursor } from "@/components/InputCursor/InputCursor";
import { Box, Text } from "@chakra-ui/react";
import { InputPhraseProp } from "./types";

export const InputPhrase: React.FC<InputPhraseProp> = ({inputStr="", correctPhrase}) => {
    const blankPhrase = correctPhrase
        .split("")
        .map((char) => (char === " " ? " " : "_"))
        .join("");
    return (
        <Box position='relative' display='inline-block'>
            <Text
                fontFamily='Courier New'
                fontSize={40}
                letterSpacing='0.75rem'
                position='absolute'
                top='0.5rem' // Adjust this value to control the height offset
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
                <InputCursor />
            </Box>
        </Box>
    );
}   