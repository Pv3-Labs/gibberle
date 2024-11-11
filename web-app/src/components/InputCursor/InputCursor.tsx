import { Text } from "@chakra-ui/react";
import { InputCursorProp } from "./types";

export const InputCursor: React.FC<InputCursorProp> = ({cursorStr="_", isVisible=true}) => {
    return (
        <Text
            fontFamily="'Hack', monospace"
            fontSize={{ base: '20px', md: '2.5vw', lg: '3vw' }}
            textAlign='center'
            letterSpacing='0.75rem'
            // following arguments are for the flashing cursor animation
            // the animation only happens when isVisible is true
            bgGradient= {isVisible ? "linear(to-r, #E18D6F, #D270BC)" : "none"}
            bgClip={isVisible ? "text" : "none"}
            color="transparent" // allows for gradient
            sx={{
                animation: isVisible ? `blink 1060ms step-end infinite` : "none",
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
