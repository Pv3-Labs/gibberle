import { Key } from "@/components/Key/Key";
import { Box, HStack, VStack } from "@chakra-ui/react";

export const Keyboard = () => {
  return (
    <Box
      w='850px'
      h='360px'
      borderRadius={'36px'}
      background={`linear-gradient(rgba(0, 0, 0, 0.33), rgba(0, 0, 0, 0.33)), #A199CA`}
      backgroundBlendMode="multiply"
      display="flex"
      justifyContent="center"
      alignItems="center"
    >
      <VStack align={'center'}>
        <HStack>
          <Key keyChar="Q" />
          <Key keyChar="W" />
          <Key keyChar="E" />
          <Key keyChar="R" />
          <Key keyChar="T" />
          <Key keyChar="Y" />
          <Key keyChar="U" />
          <Key keyChar="I" />
          <Key keyChar="O" />
          <Key keyChar="P" />
        </HStack>
        <HStack>
          <Key keyChar="A" />
          <Key keyChar="S" />
          <Key keyChar="D" />
          <Key keyChar="F" />
          <Key keyChar="G" />
          <Key keyChar="H" />
          <Key keyChar="J" />
          <Key keyChar="K" />
          <Key keyChar="L" />
        </HStack>
        <HStack>
          <Key keyChar="Z" />
          <Key keyChar="X" />
          <Key keyChar="C" />
          <Key keyChar="V" />
          <Key keyChar="B" />
          <Key keyChar="N" />
          <Key keyChar="M" />
          <Key keyChar="Ent" KeyWidth="87px" />
        </HStack>
        <Key keyChar="Space" KeyWidth="481px" />
      </VStack>
    </Box>
  );
};
