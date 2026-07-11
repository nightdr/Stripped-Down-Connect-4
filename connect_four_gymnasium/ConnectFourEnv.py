import gymnasium
from gymnasium import spaces
import pygame
import numpy as np
import time


class ConnectFourEnv(gymnasium.Env):
    metadata = {"render_modes": ["human", "rgb_array"], "fps": 3}
    WIN_REWARD = 1
    player_1_color = (224, 209, 18)
    player_2_color = (197, 7, 17)
    placeholder_color = (0, 255, 0)
    MIN_INDEX_TO_PLAY = 0
    INVALID_player = 0
    INVALID_opponent = 0

    PLACEHOLDER_PLAYER = -2
    BOARD_PADDING = 32
    COLUMN_SPACING = 4
    CIRCLE_RADIUS = 32

    def change_opponent(self, opponent):
        self._opponent = opponent

    def __init__(
        self,
        opponent=None,
        render_mode=None,
        first_player=None,
        main_player_name='IA',
        row_count: int = 6,
        col_count: int = 7,
    ):
        self._opponent = opponent  # Define the opponent

        self.row_count = row_count
        self.col_count = col_count

        # Define the action and observation spaces
        self.action_space = spaces.Discrete(self.col_count)

        self.observation_space = spaces.Box(
            low=-1,
            high=1,
            shape=(self.row_count, self.col_count),
            dtype=np.int8,
        )

        # Check if the render mode is valid
        assert render_mode is None or render_mode in self.metadata["render_modes"]
        assert first_player is None or first_player in [1, -1]
        self.render_mode = render_mode
        self.last_render_time = None
        self.window = None
        self.first_player = first_player
        self.next_player_to_play = 1
        self.main_player_name = main_player_name

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.board = np.zeros((self.row_count, self.col_count), dtype=np.int8)
        if options is not None and "starting_board" in options:
            self.board = options["starting_board"].astype(np.int8)
        assert self.board.shape == (self.row_count, self.col_count)
        self.last_move_row = None
        self.last_move_col = None
        self.invalid_move_has_been_played = False

        self._render_for_human()

        if self.first_player is None:
            self.next_player_to_play = np.random.choice([1, -1])
        else:
            other_player = -1 if self.first_player == 1 else 1
            self.next_player_to_play = (
                self.first_player if self.board.sum() == 0 else other_player
            )

        if self._opponent is not None:
            if self.next_player_to_play == -1:
                opponent_action = self._opponent.play(self.board)
                self.play_action(opponent_action)
                self.switch_player()

            self._render_for_human()

        return self.board, {}

    def _render_for_human(self):
        if self.render_mode == "human":
            self._render_frame()

    def is_column_full(self, column):
        return self.board[0, column] != 0

    def is_action_valid(self, action):
        return action >= self.MIN_INDEX_TO_PLAY and action < self.col_count and not self.is_column_full(action)

    def is_finish(self):
        if self.invalid_move_has_been_played:
            if self.last_move_col is None or self.last_move_row is None:
                return 0.0, True
            return (
                (
                    self.board[self.last_move_row, self.last_move_col]
                    .astype(np.float32)
                    .item()
                ),
                True,
            )

        if self.last_move_col is None or self.last_move_row is None:
            return 0.0, False

        if self.check_win_around_last_move(self.last_move_row, self.last_move_col):
            return (
                (
                    self.board[self.last_move_row, self.last_move_col]
                    .astype(np.float32)
                    .item()
                ),
                True,
            )

        if self.board_is_full():
            return 0.0, True

        return 0.0, False

    def play_action(self, action):
        if not self.is_action_valid(action):
            if self.render_mode == "human":
                print("action_invalid played!")
            self.invalid_move_has_been_played = True
            return

        for i in range(self.row_count - 1, -1, -1):
            if self.board[i, action] == 0:
                self.board[i, action] = self.next_player_to_play
                self.last_move_row = i
                self.last_move_col = action
                return

    def board_is_full(self):
        return np.all(self.board != 0)

    def inverse_player_position(self):
        self.board = -self.board

    def switch_player(self):
        self.next_player_to_play = -1*self.next_player_to_play

    def get_valid_actions(self):
        valid_actions = []
        for col in range(self.col_count):
            if not self.is_column_full(col):
                valid_actions.append(col)
        return valid_actions

    def clone(self):
        new_env = ConnectFourEnv(opponent=self._opponent, render_mode=self.render_mode, first_player=self.first_player)
        new_env.next_player_to_play = self.next_player_to_play
        new_env.board = self.board.copy()
        new_env.last_move_row = self.last_move_row
        new_env.last_move_col = self.last_move_col
        new_env.invalid_move_has_been_played = self.invalid_move_has_been_played
        return new_env

    def step(self, action, play_opponent=True):
        action = action.item() if isinstance(action, np.ndarray) else action

        self.play_action(action)

        result, is_finish = self.is_finish()

        self.switch_player()

        self._render_for_human()

        if is_finish and self.render_mode == "human":
            print("Finish!")

        if is_finish:
            return self.board, result, True, False, {}

        if  play_opponent and self._opponent is not None:
            opponent_action = self._opponent.play(self.board)
            opponent_result = self.step(opponent_action, play_opponent=False)
            return self.board,-1* opponent_result[1], opponent_result[2], opponent_result[3], opponent_result[4]

        return self.board, 0, False, False, {}

    def check_win_around_last_move(self, row, col):

        player = self.board[row, col]
        directions = [
            (1, 0),  # horizontal
            (0, 1),  # vertical
            (1, 1),  # diagonal /
            (1, -1)  # diagonal \
        ]

        for dr, dc in directions:
            count = 0
            for step in range(-3, 4):
                r, c = row + step * dr, col + step * dc
                if 0 <= r < self.row_count and 0 <= c < self.col_count and self.board[r, c] == player:
                    count += 1
                    if count == 4:
                        return True
                else:
                    count = 0

        return False

    def render(self):
        if self.render_mode == "rgb_array":
            return self._render_frame()

    def _render_frame(self):
        padding = 32
        padding_center = 6
        circle_radius = 32
        text_players_size = 90

        windows_width = (padding * 2) + (circle_radius * 2 * self.col_count) + padding_center * (self.col_count - 1)
        end_height_board = (padding * 2) + (circle_radius * 2 * self.row_count) + padding_center * (self.row_count - 1)
        windows_height = end_height_board + text_players_size

        pygame.font.init()

        if self.window is None and self.render_mode == "human":
            pygame.init()
            pygame.display.init()
            self.window = pygame.display.set_mode((windows_width, windows_height))

        canvas = pygame.Surface((windows_width, windows_height))
        canvas.fill((6, 66, 238))

        padding = self.BOARD_PADDING
        padding_center = self.COLUMN_SPACING
        circle_radius = self.CIRCLE_RADIUS
        i_position = padding
        for i in range(self.row_count):
            j_position = padding
            for j in range(self.col_count):
                color = (245, 245, 245)
                if self.board[i, j] == self.first_player:
                    color = self.player_1_color
                elif self.board[i, j] == -1 * self.first_player:
                    color = self.player_2_color
                elif self.board[i, j] == self.PLACEHOLDER_PLAYER:
                    color = self.placeholder_color
                pygame.draw.circle(canvas, color, (j_position + circle_radius, i_position + circle_radius), circle_radius)
                j_position += (circle_radius * 2) + padding_center
            i_position += (circle_radius * 2) + padding_center

        # Display opponent's color and name
        text_position_y_first_player = end_height_board
        pygame.draw.circle(canvas, self.player_2_color, (50, text_position_y_first_player + circle_radius / 4),
                           circle_radius / 2)
        font = pygame.font.Font(None, 36)
        opponent_name =  self._opponent.getName() if self._opponent is not None else "Second player"
        text = font.render(f"{opponent_name}", 1, (10, 10, 10))

        canvas.blit(text, (80, text_position_y_first_player))

        # Display user's color
        text_position_y_second_player = text_position_y_first_player + 40
        pygame.draw.circle(canvas, self.player_1_color, (50, text_position_y_second_player + circle_radius / 4),
                           circle_radius / 2)
        text = font.render(self.main_player_name, 1, (10, 10, 10))
        canvas.blit(text, (80, text_position_y_second_player))

        if self.render_mode == "human":
            self.window.blit(canvas, canvas.get_rect())
            pygame.event.pump()
            pygame.display.update()
        else:
            return np.transpose(pygame.surfarray.array3d(canvas), (1, 0, 2))

    def poll_click_action(self) -> int | None:
        """Display a placeholder circle where the player's mouse is, and
        optionally return the column index they clicked in.

        Injects a PLACEHOLDER_PLAYER entry into the board before rendering to
        display a placeholder circle for the available circle in the column
        where the mouse is. Removes said entry after the render finishes but
        before returning.
        """

        # to make the first and last columns have the same width as the rest
        padding = self.BOARD_PADDING - int(self.COLUMN_SPACING // 2)
        for event in pygame.event.get():
            if event.type in [pygame.MOUSEMOTION, pygame.MOUSEBUTTONDOWN]:
                mouse_x, _ = pygame.mouse.get_pos()

                col_width = self.COLUMN_SPACING + self.CIRCLE_RADIUS * 2
                col_i = (mouse_x - padding) // col_width
                col_i = max(col_i, 0)
                col_i = min(col_i, self.col_count - 1)

                # find which row has an empty space, if any
                row_i = -1
                for i in range(self.board.shape[0] - 1, -1, -1):
                    if self.board[i, col_i] == 0:
                        row_i = i
                        break

                # inject a placeholder player circle
                if row_i != -1:
                    assert self.board[row_i, col_i] == 0
                    self.board[row_i, col_i] = self.PLACEHOLDER_PLAYER

                self._render_frame()

                # revert the placeholder player circle in the board to not
                # affect game logic
                if row_i != -1:
                    self.board[row_i, col_i] = 0

                # return the column index if the player clicked, and the column
                # isn't full
                if event.type == pygame.MOUSEBUTTONDOWN and row_i != -1:
                    return col_i
                return None
        return None

    def close(self):
        if self.window is not None:
            pygame.display.quit()
            pygame.quit()