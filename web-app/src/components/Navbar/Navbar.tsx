import {
  Heading,
  HStack,
  Icon,
  IconButton,
  Tooltip,
  useMediaQuery,
} from "@chakra-ui/react";
import { HiOutlineQuestionMarkCircle } from "react-icons/hi";
import { IoSettingsOutline } from "react-icons/io5";
import { MdLeaderboard, MdLightbulbOutline } from "react-icons/md";
import { NavbarProp } from "./types";

export default function Navbar({
  onHintClick,
  onTutorialClick,
  onStatsClick,
  onSettingsClick,
  // theme,
  disabled = false,
}: NavbarProp) {
  const [isMobile] = useMediaQuery("(max-width: 768px)");

  return (
    <>
      <HStack
        justifyContent="flex-end"
        alignItems="center"
        width="100%"
        padding={"1rem"}
        spacing={isMobile ? "1rem" : "1.5rem"}
        paddingRight={"2rem"}
        paddingBottom={isMobile ? "0" : "1rem"}
        position="relative"
        zIndex={1}
      >
        <Tooltip label="Hint" aria-label="Hint" bg="#292929">
          <IconButton
            icon={
              <Icon
                as={MdLightbulbOutline}
                boxSize="1.5rem"
                // bgGradient="linear(to-r, #E18D6F, #D270BC)"
                // bgClip="text"
                // fill="transparent"
              />
            }
            onClick={onHintClick}
            aria-label="Show Hint"
            // isDisabled={disabled}
            color="white"
            bg="transparent"
            _hover={{ bg: "transparent" }}
            _active={{ bg: "transparent" }}
            _focus={{ boxShadow: "none" }}
            size={"lg"}
            p={0}
            minW={0}
          />
        </Tooltip>
        <Tooltip label="Tutorial" aria-label="Tutorial" bg="#292929">
          <IconButton
            icon={<Icon as={HiOutlineQuestionMarkCircle} boxSize={"1.5rem"} />}
            onClick={onTutorialClick}
            aria-label="Show Tutorial"
            isDisabled={disabled}
            color="white"
            bg="transparent"
            _hover={{ bg: "transparent" }}
            _active={{ bg: "transparent" }}
            _focus={{ boxShadow: "none" }}
            p={0}
            minW={0}
          />
        </Tooltip>
        <Tooltip label="Stats" aria-label="Stats" bg="#292929">
          <IconButton
            icon={<Icon as={MdLeaderboard} boxSize={"1.5rem"} />}
            onClick={onStatsClick}
            aria-label="Show Stats"
            isDisabled={disabled}
            color="white"
            bg="transparent"
            _hover={{ bg: "transparent" }}
            _active={{ bg: "transparent" }}
            _focus={{ boxShadow: "none" }}
            p={0}
            minW={0}
          />
        </Tooltip>
        <Tooltip label="Settings" aria-label="Settings" bg="#292929">
          <IconButton
            icon={<Icon as={IoSettingsOutline} boxSize={"1.5rem"} />}
            onClick={onSettingsClick}
            aria-label="Show Settings"
            isDisabled={disabled}
            color="white"
            bg="transparent"
            _hover={{ bg: "transparent" }}
            _active={{ bg: "transparent" }}
            _focus={{ boxShadow: "none" }}
            p={0}
            minW={0}
          />
        </Tooltip>
      </HStack>
      <Heading
        bgGradient="linear(to-r, #E18D6F, #D270BC)"
        bgClip="text"
        fontWeight="bold"
        fontSize={"6vh"}
        textAlign="center"
        position="relative"
        mt={isMobile ? "0rem" : "1.5rem"}
      >
        Gibberle
      </Heading>
    </>
  );
}
