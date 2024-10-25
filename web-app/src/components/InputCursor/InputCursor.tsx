import { Text } from "@chakra-ui/react";
import { InputCursorProp } from "./types";

export const InputCursor: React.FC<InputCursorProp> = ({cursorStr="_"}) => {
    return (
        <Text
            fontFamily='Roboto Mono'
            fontSize={50}
            align='left'
            letterSpacing='0.75rem'
            // following arguments are for the flashing cursor animation
            bgGradient="linear(to-r, #E18D6F, #D270BC)"
            bgClip="text"
            color="transparent" // allows for gradient
            sx={{
                animation: `blink 1060ms step-end infinite`,
                "@keyframes blink": {
                    "0%, 100%": { opacity: 1 }, // changes cursor color to gradient
                    "50%": { opacity: 0 }, // changes cursor back to text color
                }
            }}
        >
            {cursorStr}
        </Text>
    );
}
