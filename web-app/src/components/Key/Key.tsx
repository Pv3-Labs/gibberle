import { KeyProp } from '@/components/Key/types';
import { Button, Text } from '@chakra-ui/react';

export const Key: React.FC<KeyProp> = ({keyChar, KeyWidth='72px'}) => {
  return (
    <Button 
      bg='#1C1C1C' 
      w={KeyWidth}
      h='72px'
      borderRadius={12}
    >
      <Text fontSize='4xl' fontWeight='medium'>
        {keyChar}
      </Text>
    </Button>
  );
};
