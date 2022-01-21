import collections
import time
import numpy as np
from agent import Agent
from experience import ExperienceReplay


class Game:
    def __init__(self):
        self.wait = 0
        self.field = [[0] * SIZE for _ in range(SIZE)]
        self.directions = [(0, 0), (-1, 0), (-1, 1), (0, 1), (1, 1), (1, 0), (1, -1), (0, -1), (-1, -1)]

        self.monsters = []
        self.bullets = []

        self.player = Player(SIZE // 2, SIZE // 2, self)
        self.player.update()

        self.last_reward = 1
        self.shoot_reward = 0
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
        x = int(self.player.x + 0.5)
        y = int(self.player.y + 0.5)
        x /= SIZE
        y /= SIZE
        if x > 0.5:
            x = 1 - x
        if y > 0.5:
            y = 1 - y

        self.last_reward = 1
        if min(x, y) < 0.3:
            self.last_reward = 0

        self.shoot_reward = 0
        self.done = 0
        self.clear()

        # 0 1 2
        # 3 4 5
        # 6 7 8

        # action = move_action + 9 * shooting_action

        self.player.move(action % 9)
        self.player.spawn_bullet(action // 9)

        for monster in self.monsters:
            monster.move()
        for bullet in self.bullets:
            bullet.move()

        self.wait += 1
        if self.wait < 3:
            pass
        else:
            spawn_wave = np.random.binomial(n=1, p=0.1)
            if spawn_wave:
                for _ in range(9):
                    should_spawn = np.random.binomial(n=1, p=0.8)
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
        self.wait = 0
        self.done = 0
        self.last_reward = 1
        self.shoot_reward = 0

    def restart(self):
        self.done = 1
        self.last_reward = -50
        self.shoot_reward = -50
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
        # arr = [0] * (SIZE * SIZE * 3)
        arr = np.zeros((3, SIZE, SIZE))

        x = int(self.player.x + 0.5)
        y = int(self.player.y + 0.5)
        # arr[y + SIZE * x] = 1
        # arr[y + 1 + SIZE * x] = 1
        # arr[y + 1 + SIZE * (x + 1)] = 1
        # arr[y + SIZE * (x + 1)] = 1
        arr[0][y][x] = 1
        arr[0][y + 1][x] = 1
        arr[0][y + 1][x + 1] = 1
        arr[0][y][x + 1] = 1

        for bullet in self.bullets:
            x = int(bullet.x + 0.5)
            y = int(bullet.y + 0.5)
            # arr[SIZE * SIZE + y + SIZE * x] = 1
            arr[1][y][x] = 1

        for monster in self.monsters:
            x = int(monster.x + 0.5)
            y = int(monster.y + 0.5)
            # arr[SIZE * SIZE * 2 + y + SIZE * x] = 1
            # arr[SIZE * SIZE * 2 + y + 1 + SIZE * x] = 1
            # arr[SIZE * SIZE * 2 + y + 1 + SIZE * (x + 1)] = 1
            # arr[SIZE * SIZE * 2 + y + SIZE * (x + 1)] = 1
            arr[2][y][x] = 1
            arr[2][y + 1][x] = 1
            arr[2][y + 1][x + 1] = 1
            arr[2][y][x + 1] = 1

        return arr


class Creature:
    def __init__(self, x, y, _game):
        self.x = x
        self.y = y
        self.game = _game


class Player(Creature):
    def move(self, _action):
        self.x += self.game.directions[_action][0]
        self.y += self.game.directions[_action][1]

        self.x = min(SIZE - 2, self.x)
        self.y = min(SIZE - 2, self.y)
        self.x = max(0, self.x)
        self.y = max(0, self.y)

        for monster in self.game.monsters:
            if self.check_collide(monster):
                self.game.restart()
                return

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
    def __init__(self, x, y, direction, _game):
        direction += 1
        self.x = x
        self.y = y
        self.game = game
        self.direction = direction
        if 2 <= direction <= 4:
            self.y += 1
        if 4 <= direction <= 6:
            self.x += 1
        self.game.field[self.x][self.y] = 1

    def move(self):
        old_x = self.x
        old_y = self.y

        self.x += 3 * game.directions[self.direction][0]
        self.y += 3 * game.directions[self.direction][1]

        x = self.x
        y = self.y

        while old_x != x or old_y != y:
            for monster in self.game.monsters:
                if self.intersects_monster(monster, old_x, old_y):
                    self.game.shoot_reward += 5

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
            self.game.shoot_reward -= 1  # TODO
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
        elif np.random.binomial(n=1, p=0.75):  # SHOULD CHASE
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


FREQUENCY = 4
BATCH_SIZE = 32
BUFFER_SIZE = 65536
REPLAY_START_SIZE = 1024

SIZE = 32
N_FRAMES = 1      # state = 2 last frames
N_ACTIONS = 9 * 9     # (8 directions to shoot or not shoot lol) * (8 directions to go or stay lol)
N_FEATURES = SIZE * SIZE * 3  # player, player's bullets, enemies # DEPRECATED

TRAIN = False  # todo
COMPUTER = True
SHOULD_EXECUTE = True

DELTA_EPSILON = 0.99995

MODEL_PATH = "model1"
MODEL_PATH2 = "model2"

game = Game()

last_s = collections.deque(maxlen=N_FRAMES)
next_s = collections.deque(maxlen=N_FRAMES)

agent = Agent(N_FRAMES, N_FEATURES, N_ACTIONS, 1 * TRAIN, DELTA_EPSILON, 0.1 * TRAIN, 0.95, 1e-4, MODEL_PATH, None)
# shooting = Agent(N_FRAMES, N_FEATURES, 8, 1 * TRAIN, DELTA_EPSILON, 0.1 * TRAIN,
# 0.95, 1e-4, MODEL_PATH2, [N_FEATURES * N_FRAMES, 4096, 2048, 1024, 512, 256])

buffer = ExperienceReplay(BUFFER_SIZE, N_FRAMES)
Experience = collections.namedtuple('Experience', field_names=['state', 'action', 'reward', 'next_state', 'done'])

action = 0
# shooting_action = 0
sum_r = 0
max_t = 100
t = 0
avg_steps = 0
best_r = 0
temp_r = 0
last_time = time.time()

while SHOULD_EXECUTE:
    state = game.observations()
    # print(len(state[0]))

    # for i in range(3 * SIZE):
    #     for j in range(SIZE):

    reward = game.last_reward
    shooting_reward = game.shoot_reward
    done = game.done

    next_s.append(state)

    if TRAIN:
        if len(buffer) < REPLAY_START_SIZE:
            pass
        elif done and t % FREQUENCY == 0:
            batch = buffer.sample(BATCH_SIZE)
            states, actions, shooting_actions, rewards, shooting_rewards, next_states, is_done = batch
            agent.learn(states, actions, rewards, next_states, is_done)  # TODO
            # shooting.learn(states, shooting_actions, shooting_rewards, next_states, is_done)

        if len(last_s) == N_FRAMES:
            buffer.append(Experience(last_s[-1], action, reward, next_s[-1], done))  # todo np.concatenate(next_s)
            sum_r += reward + shooting_reward
            temp_r += reward + shooting_reward

    last_s.append(next_s[-1])

    if len(last_s) == N_FRAMES:
        action = agent.get_action(last_s[-1])  # TODO
        # shooting_action = shooting.get_action(last_s[-1])  # TODO few frames
        # print(action)
    if COMPUTER:
        if not done:
            game.step(action)
            # game.player.spawn_bullet(shooting_action)
            if not TRAIN:
                time.sleep(0.5)
        else:
            action = 0
            shooting_action = 0
            last_s.clear()
            next_s.clear()
            avg_steps += game.wait
            best_r = max(best_r, temp_r)
            temp_r = 0
            game.launch()

    if TRAIN and done:
        t += 1
        if t % max_t == 0:
            agent.save()  # TODO
            # shooting.save()
            print(f"t = {t}, mean sum = {sum_r / max_t:.2f}, best score = "
                  f"{best_r:.2f}, average steps = {avg_steps / max_t:.2f}, time = {(time.time() - last_time):.2f}s")
            avg_steps = 0
            sum_r = 0
            best_r = -100
            last_time = time.time()
        # if t > 100000:
        #     buffer.clear()
