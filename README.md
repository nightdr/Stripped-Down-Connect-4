## Environment Details

In this section, we provide an overview of the Connect Four environment, including the action space, observation space, rewards, and episode termination conditions.

### Action Space

The action space in the Connect Four environment is discrete, with a total of 7 possible actions. Each action corresponds to a column in the game board where a player can drop their piece. The action space is represented as follows:

```python
self.action_space = spaces.Discrete(self.COLUMNS_COUNT)
```

### Observation Space

The observation space in the Connect Four environment is a 2D array representing the game board. Each cell in the array can have one of three possible values: 0 (empty), 1 (player's piece), or -1 (opponent's piece). The observation space is defined as follows:

```python
self.observation_space = spaces.Box(low=-1, high=1, shape=(self.ROWS_COUNT, self.COLUMNS_COUNT), dtype=np.int32)
```

### Rewards

The reward system in the Connect Four environment is designed to encourage the AI model to learn how to win the game. The rewards are as follows:

- A reward of +1 is given when the AI model wins the game by connecting four of its pieces in a row, either horizontally, vertically, or diagonally.
- A reward of -1 is given when the AI model loses the game or plays an invalid action.
- A reward of 0 is given for all other actions that do not result in a win or loss.

### Episode Termination

An episode in the Connect Four environment terminates under the following conditions:

- The player wins the game by connecting four of its pieces in a row, either horizontally, vertically, or diagonally.
- The player loses the game by allowing the opponent to connect four of their pieces in a row, or by playing an invalid action.
- The game board is completely filled, resulting in a draw.

## Detailed Documentation

For a comprehensive overview of the methods and classes used in this environment, please refer to the [detailed documentation](https://github.com/lucasBertola/Connect-4-Gym-env-Reinforcement-learning/blob/main/DOCUMENTATION.md) available on the GitHub repository. This documentation provides in-depth explanations and examples to help you better understand the inner workings of the Connect Four environment and make the most of its features.

## Testing

We believe in the importance of testing. That's why we have included a suite of tests in the `test` directory. To run the tests, simply use the command `pytest`.

## License

This project is licensed under the MIT License. See the `LICENSE` file for details.
