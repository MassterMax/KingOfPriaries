import collections
import time
import numpy as np
from agent import Agent
from experience import ExperienceReplay


class Game:
    def __init__(self):
        self.field = [[0] * SIZE for _ in range(SIZE)]
        self.directions = [(0, 0), (-1, 0), (-1, 1), (0, 1), (1, 1), (1, 0), (1, -1), (0, -1), (-1, -1)]

        self.monsters = []
        self.bullets = []

        self.player = Player(SIZE // 2, SIZE // 2, self)
        self.player.update()

        self.last_reward = 1
        self.done = 0

    def print_field(self):
        print("\n"*SIZE)
        for i in range(SIZE):
            for j in range(SIZE):
                if self.field[i][j] == 0:
                    print('.', end=" ")
                if self.field[i][j] == 1:
                    print('P', end=" ")
                if self.field[i][j] == 2:
                    print('*', end=" ")
                if self.field[i][j] == 3:
                    print('M', end=" ")

            print()

    def step(self, action):
        self.last_reward = 1
        self.done = 0
        self.clear()
        self.player.move(action)
        for monster in self.monsters:
            monster.move()
        for bullet in self.bullets:
            bullet.move()

        spawn_wave = np.random.binomial(n=1, p=0.2)
        if spawn_wave:
            for _ in range(5):
                x = -1
                y = -1
                should_spawn = np.random.binomial(n=1, p=0.7)
                if should_spawn:

                    x = 0 if np.random.binomial(1, 0.5) else SIZE - 2
                    y = np.random.randint(SIZE // 3, 2 * SIZE // 3)
                    if np.random.binomial(1, 0.5):
                        x, y = y, x

                    for monster in self.monsters:
                        if abs(monster.x - x) < 2 and abs(monster.y - y) < 2:
                            should_spawn = False
                            break
                    if should_spawn:
                        m = Monster(x, y, self)
                        m.times = 0
                        self.monsters.append(m)
                        self.field[x][y] = 3
                        self.field[x + 1][y] = 3
                        self.field[x][y + 1] = 3
                        self.field[x + 1][y + 1] = 3

        if not TRAIN:
            self.print_field()

    def clear(self):
        # print('\n' * 80)
        for i in range(SIZE):
            for j in range(SIZE):
                self.field[i][j] = 0

    def launch(self):
        self.done = 0
        self.last_reward = 1

    def restart(self):
        self.done = 1
        self.last_reward = -100
        self.clear()
        self.monsters.clear()
        self.bullets.clear()
        extra_start = np.random.binomial(1, 0.2)
        if extra_start:
            x = SIZE // 3 + (SIZE // 3) * np.random.binomial(1, 0.5)
            y = SIZE // 3 + (SIZE // 3) * np.random.binomial(1, 0.5)
            self.player = Player(x, y, self)
        else:
            self.player = Player(SIZE // 2, SIZE // 2, self)
        self.player.update()
        # self.print_field()

    def observations(self):
        arr = [0] * N_FEATURES

        x = int(self.player.x)
        y = int(self.player.y)
        arr[y + SIZE * x] = 1
        arr[y + SIZE * (x + 1)] = 1
        arr[y + 1 + SIZE * x] = 1
        arr[y + 1 + SIZE * (x + 1)] = 1

        for bullet in self.bullets:
            x = int(bullet.x)
            y = int(bullet.y)
            arr[SIZE * SIZE + y + SIZE * x] = 1

        for monster in self.monsters:
            x = int(monster.x)
            y = int(monster.y)
            arr[SIZE * SIZE * 2 + y + SIZE * x] = 1
            arr[SIZE * SIZE * 2 + y + SIZE * (x + 1)] = 1
            arr[SIZE * SIZE * 2 + y + 1 + SIZE * x] = 1
            arr[SIZE * SIZE * 2 + y + 1 + SIZE * (x + 1)] = 1

        return arr


class Creature:
    def __init__(self, x, y, game):
        self.x = x
        self.y = y
        self.game = game


class Player(Creature):
    def move(self, action):
        if action < 8:
            action += 1
            self.x += self.game.directions[action][0]
            self.y += self.game.directions[action][1]

        # if x < 0 or y < 0 or SIZE - 2 < x or SIZE - 2 < y:
        #     self.game.restart()
        #     return
            self.x = min(SIZE - 2, self.x)
            self.y = min(SIZE - 2, self.y)
            self.x = max(0, self.x)
            self.y = max(0, self.y)

            for monster in self.game.monsters:
                if self.check_collide(monster):
                    self.game.restart()
                    return

            direction = (action + 3) % 8 + 1
            self.spawn_bullet(direction)
        else:
            action -= 8
            action += 1
            direction = (action + 3) % 8 + 1
            self.spawn_bullet(direction)

        self.update()

    def update(self):
        self.game.field[self.x][self.y] = 1
        self.game.field[self.x + 1][self.y] = 1
        self.game.field[self.x][self.y + 1] = 1
        self.game.field[self.x + 1][self.y + 1] = 1

    def spawn_bullet(self, direction):
        bullet = Bullet(self.x, self.y, direction, self.game)
        game.bullets.append(bullet)

    def check_collide(self, monster):
        return abs(self.x - monster.x) < 2 and abs(self.y - monster.y) < 2


class Bullet:
    def __init__(self, x, y, direction, game):
        self.x = x
        self.y = y
        self.game = game
        self.direction = direction
        if 2 <= direction <= 4:
            self.y += 1
        if 4 <= direction <= 6:
            self.x += 1

    def move(self):
        old_x = self.x
        old_y = self.y

        self.x += 3 * game.directions[self.direction][0]
        self.y += 3 * game.directions[self.direction][1]

        x = self.x
        y = self.y

        # print(old_x, old_y, x, y, sep=" ")
        while old_x != x or old_y != y:
            for monster in self.game.monsters:
                # print(monster.x, monster.y, old_x, old_y, sep = " ")
                if self.intersects_monster(monster, old_x, old_y):
                    self.game.last_reward += 5  # REWARD CHANGED

                    self.game.field[monster.x][monster.y] = 0
                    self.game.field[monster.x + 1][monster.y] = 0
                    self.game.field[monster.x][monster.y + 1] = 0
                    self.game.field[monster.x + 1][monster.y + 1] = 0

                    game.monsters.remove(monster)
                    game.bullets.remove(self)
                    return
            old_x += game.directions[self.direction][0]
            old_y += game.directions[self.direction][1]

        if x < 0 or y < 0 or SIZE - 1 < x or SIZE - 1 < y:
            game.bullets.remove(self)
            # self.game.last_reward -= 1  # PENALTY
        else:
            self.game.field[x][y] = 2

    @staticmethod
    def intersects_monster(monster, old_x, old_y):
        return monster.x == old_x and monster.y == old_y or monster.x + 1 == old_x and monster.y == old_y or monster.x == old_x and monster.y + 1 == old_y or monster.x + 1 == old_x and monster.y + 1 == old_y


class Monster(Creature):
    def move(self):
        old_x = self.x
        old_y = self.y
        if self.times < SIZE // 3:
            if old_x < SIZE // 3:
                dx = 1
            elif old_x > 2 * SIZE // 3:
                dx = -1
            else:
                dx = 0

            if old_y < SIZE // 2:
                dy = 1
            elif old_y > 2 * SIZE // 3:
                dy = -1
            else:
                dy = 0

            self.x += dx
            self.y += dy

            self.times += 1
        elif np.random.binomial(n=1, p=0.6):
            x = self.game.player.x
            y = self.game.player.y

            min_dist = 1000000000
            act = 0
            i = 0

            for d in self.game.directions:
                new_x = self.x + d[0]
                new_y = self.y + d[1]
                if self.dist(new_x, x, new_y, y) < min_dist:
                    min_dist = self.dist(new_x, x, new_y, y)
                    act = i
                i += 1

            self.x += self.game.directions[act][0]
            self.y += self.game.directions[act][1]
        else:
            act = self.game.directions[np.random.choice(len(self.game.directions), 1)[0]]

            self.x += act[0]
            self.y += act[1]

        self.x = min(self.x, SIZE - 2)
        self.y = min(self.y, SIZE - 2)
        self.x = max(self.x, 0)
        self.y = max(self.y, 0)

        intersects = False
        for monster in self.game.monsters:
            if monster != self and self.dist(monster.x, self.x, monster.y, self.y) < 3:
                self.x = old_x
                self.y = old_y
                if intersects:
                    break

                intersects = True
                while self.dist(monster.x, self.x, monster.y, self.y) < 3:
                    act = self.game.directions[np.random.choice(len(self.game.directions), 1)[0]]
                    self.x = old_x + act[0]
                    self.y = old_y + act[1]

        if self.game.player.check_collide(self):
            self.game.restart()
            return

        self.game.field[self.x][self.y] = 3
        self.game.field[self.x + 1][self.y] = 3
        self.game.field[self.x][self.y + 1] = 3
        self.game.field[self.x + 1][self.y + 1] = 3

    @staticmethod
    def dist(x1, x2, y1, y2):
        return (x1 - x2) ** 2 + (y1 - y2) ** 2


BATCH_SIZE = 500
BUFFER_SIZE = 50000
REPLAY_START_SIZE = 4000

N_FRAMES = 4      # state = 4 frames
N_ACTIONS = 16     # 8 directions to shoot * 2 (stay/go in opposite direction)
SIZE = 13
N_FEATURES = 3 * SIZE**2  # player, player's bullets, enemies

TRAIN = True
COMPUTER = True
SHOULD_EXECUTE = True

MODEL_PATH = "C:/Users/maxma/Desktop/real курсач/callCS/config2"

game = Game()

last_s = collections.deque(maxlen=N_FRAMES)
next_s = collections.deque(maxlen=N_FRAMES)

agent = Agent(N_FRAMES, N_FEATURES, N_ACTIONS, 1 * TRAIN, 0.999995, 0.1 * TRAIN, 0.9, 1e-4, MODEL_PATH)

buffer = ExperienceReplay(BUFFER_SIZE, N_FRAMES)
Experience = collections.namedtuple('Experience', field_names=['state', 'action', 'reward', 'next_state', 'done'])

action = 0
sum_r = 0
max_t = 100
t = 0

while SHOULD_EXECUTE:
    state = game.observations()
    reward = game.last_reward
    done = game.done

    # print(reward, done, sep=" ")
    # action = np.random.choice(9, 1)[0]
    # game.step(action)
    # time.sleep(0.5)

    # for i in range(96):
    #    for j in range(32):
    #        print(state[32 * i + j], end=" ")
    #    print()
    #    if (i + 1) % 32 == 0:
    #        print()

    next_s.append(state)

    if TRAIN:
        if len(buffer) < REPLAY_START_SIZE:
            pass
        elif done and t % 10 == 0:
            batch = buffer.sample(BATCH_SIZE)
            states, actions, rewards, next_states, is_done = batch
            agent.learn(states, actions, rewards, next_states, is_done)

        if len(last_s) == 4:
            buffer.append(Experience(np.concatenate(last_s), action, reward, np.concatenate(next_s), done))
            sum_r += reward

    last_s.append(next_s[-1])

    if len(last_s) == 4:
        action = agent.get_action(np.concatenate(last_s))
        # print(action)
    if COMPUTER:
        if not done:
            game.step(action)
            if not TRAIN:
                time.sleep(0.5)
        else:
            action = 0
            last_s.clear()
            next_s.clear()
            game.launch()

    if TRAIN and done:
        t += 1
        if t % max_t == 0:
            agent.save()
            print(f"t = {t}, mean sum = {sum_r / max_t}, model saved")
            sum_r = 0
