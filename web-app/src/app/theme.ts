// src/app/theme.tsx
"use client";

import { extendTheme } from "@chakra-ui/react";

const colors = {
    background: '#1C1C1C',
    primary: '#E18D6F',
    secondary: '#D270BC',
    accent: '#A199CA',
    text: '#EAEAEA',
};

export const theme = extendTheme({
    colors: {
        ...colors,
    },
    styles: {
        global: {
            body: {
                backgroundColor: colors.background,  // set global background color
                color: colors.text,  // set global text color
            },
        },
    },
});
