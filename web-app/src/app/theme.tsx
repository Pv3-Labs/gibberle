// src/app/theme.tsx
"use client";

import { extendTheme } from "@chakra-ui/react";

const colors = {
  background: "#1C1C1C",
  primary: "#E18D6F",
  secondary: "#D270BC",
  accent: "#A199CA",
  text: "#EAEAEA",
};

export const theme = extendTheme({
  colors: {
    ...colors,
  },
  styles: {
    global: {
      body: {
        backgroundColor: colors.background,
        color: colors.text,
      },
      "@font-face": {
        fontFamily: "Hack",
        src: 'url("/fonts/Hack-Regular.ttf") format("truetype")',
        fontWeight: "normal",
        fontStyle: "normal",
      },
    },
  },
});
