// app/layout.tsx

"use client";

import { Box } from "@chakra-ui/react";

export default function RootLayout({
  children,
}: {
  children: React.ReactNode;
}) {
  return (
    <Box bg="212121" color="white" minH="100vh">
      {children}
    </Box>
  );
}
