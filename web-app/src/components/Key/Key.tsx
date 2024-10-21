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
      bg='#1C1C1C'
      w={keyWidth}
      h='72px'
      borderRadius={12}
      // boxShadow is the gradient glow arund the key when pressed
      boxShadow={isPressed ? "-2px -2px 3px 1px #E18D6F, 2px 2px 3px 1px #D270BC, -2px 2px 3px 1px #E18D6F, 2px -2px 3px 1px #D270BC" : "0 0 0 0px transparent"}
      _hover={{bg: '#333333'}}
    >
      <Text
        fontSize='4xl' 
        fontWeight='medium'
        bgGradient={isPressed ? 'linear(to-r, #E18D6F, #D270BC)' : 'undefined'}
        bgClip='text'
        color={isPressed ? 'transparent' : '#FFFFFF'}
      >
        {keyChar}
      </Text>
    </Button>
  );
};
