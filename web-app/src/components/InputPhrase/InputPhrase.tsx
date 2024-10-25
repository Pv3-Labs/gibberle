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
                fontFamily='Roboto Mono'
                fontSize={50}
                align='left'
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