# Snake AI

Neural Expect Sarsa Reinforcement Learning Agent for [Snake](https://github.com/lochungtin/snake)

# Demo

This is the average performance of the agent after training for 1000 epochs

<p align='center'>
    <img src='./docs/demo.gif'>
</p>
<p align='center'>
    this gif is sped up to x2 speed
</p>

# Technical Details

## OpenCV

Before training, the program has to configure the OpenCV reading to read specific pixels to gather information for the training agent.

### Configuration Process

1. The program first captures a screenshots of the selected monitor
2. Using the precoded pixel color value, the program finds the dot in the top left corner
3. Using the dot, the program reconfigures the OpenCV capture configuration to specifically focus on the grid
4. The program then updates the other dot positions which convey additional information about the game state (more about the dots can be read in the [game's](https://github.com/lochungtin/snake) repo readme)
5. Finally, the program scans the grid and using the light levels of the 121 scanned pixels to determine the dimension of the grid. This is required as the game allows configurations of 11x11, 9x9, and also 7x7.
6. After configuring, the agent can start training with the information read from the opencv thread
