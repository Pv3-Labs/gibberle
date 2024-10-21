import { KeyProp } from '@/components/Key/types';
import { Button, Text } from '@chakra-ui/react';

export const Key: React.FC<KeyProp> = ({keyChar, keyWidth='72px', isPressed=false}) => {
  // Enter Key is different width then the rest of the keys
  // THOUGHT: should we do something where each key has a fixed padding
  //  instead of a width attribute? Or something lke it?
  //  - this would make it so we have don't have to set the width
  if (keyChar === "Enter") {
    keyWidth = "87px";
  };
  
  return (
    <Button 
      bg={isPressed ? 'FF0000' : '#1C1C1C' }
      w={keyWidth}
      h='72px'
      borderRadius={12}
    >
      <Text fontSize='4xl' fontWeight='medium'>
        {keyChar}
      </Text>
    </Button>
  );
};
