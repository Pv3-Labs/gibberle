export interface NavbarProp {
    onHintClick: () => void;
    onTutorialClick: () => void;
    onStatsClick: () => void;
    onSettingsClick: () => void;
    theme?: object;
    disabled?: boolean;
}