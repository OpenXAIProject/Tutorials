class Config:

    #########################################################################
    # Game configuration

    # Name of the game, with version (e.g. PongDeterministic-v0)
    ATARI_GAME = 'PongDeterministic-v0'

    # Enable to see the trained agent in action
    PLAY_MODE = False

    # Input of the DNN
    STACKED_FRAMES = 4
    IMAGE_WIDTH = 84
    IMAGE_HEIGHT = 84