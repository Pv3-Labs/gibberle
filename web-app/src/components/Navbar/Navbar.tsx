import { NavbarProp } from "./types";
import { Heading, IconButton, Box, HStack, Tooltip, Spacer, Icon /*useColorModeValue, Modal, useDisclosure*/ } from "@chakra-ui/react";
import { MdLightbulbOutline, MdLeaderboard} from "react-icons/md";
import { HiOutlineQuestionMarkCircle } from "react-icons/hi";
import { IoSettingsOutline } from "react-icons/io5";

export default function Navbar({
    onHintClick,
    onTutorialClick,
    onStatsClick,
    onSettingsClick,
    // theme,
    disabled = false
  }: NavbarProp) {

    return (
        <Box
            marginTop={"5vh"}
            width={"100%"}
        >
            <HStack justifyContent="space-between" alignItems="center" width="100%">
                <Spacer />
        
                <Heading
                    bgGradient="linear(to-r, #E18D6F, #D270BC)"
                    bgClip="text"
                    fontWeight="bold"
                    fontSize={"6vh"}
                    textAlign="center"
                    marginLeft={"16vw"}
                >
                    Gibberle
                </Heading>

                <Spacer />

                <HStack spacing={"2.5vw"} marginRight={"2vw"}>
                    <Tooltip label="Hint" aria-label="Hint">
                        <IconButton 
                            icon={<Icon as={MdLightbulbOutline} boxSize="1.5rem"/>}
                            onClick={onHintClick}
                            aria-label="Show Hint"
                            isDisabled={disabled}
                            bg="transparent"
                            _hover={{ bg: "transparent" }} 
                            _active={{ bg: "transparent" }} 
                            _focus={{ boxShadow: "none" }}
                            size={"lg"}
                            p={0}
                            minW={0}   
                        />
                    </Tooltip>
                    <Tooltip label="Tutorial" aria-label="Tutorial">
                        <IconButton 
                            icon={<Icon as={HiOutlineQuestionMarkCircle} boxSize={"1.5rem"} />}
                            onClick={onTutorialClick}
                            aria-label="Show Tutorial"
                            isDisabled={disabled}
                            color={"#A199CA"}
                            bg="transparent"
                            _hover={{ bg: "transparent" }} 
                            _active={{ bg: "transparent" }} 
                            _focus={{ boxShadow: "none" }}
                            p={0}
                            minW={0}   
                        />
                    </Tooltip>
                    <Tooltip label="Stats" aria-label="Stats">
                        <IconButton 
                            icon={<Icon as={MdLeaderboard} boxSize={"1.5rem"} />}
                            onClick={onStatsClick}
                            aria-label="Show Stats"
                            isDisabled={disabled}
                            color={"#A199CA"}
                            bg="transparent"
                            _hover={{ bg: "transparent" }} 
                            _active={{ bg: "transparent" }} 
                            _focus={{ boxShadow: "none" }}
                            p={0}
                            minW={0}   
                        />
                    </Tooltip>
                    <Tooltip label="Settings" aria-label="Settings">
                        <IconButton 
                            icon={<Icon as={IoSettingsOutline} boxSize={"1.5rem"} />}
                            onClick={onSettingsClick}
                            aria-label="Show Settings"
                            isDisabled={disabled}
                            color={"#A199CA"}
                            bg="transparent"
                            _hover={{ bg: "transparent" }} 
                            _active={{ bg: "transparent" }} 
                            _focus={{ boxShadow: "none" }}
                            p={0}
                            minW={0}   
                        />
                    </Tooltip>
                </HStack>
            </HStack>
        </Box>
    )
}