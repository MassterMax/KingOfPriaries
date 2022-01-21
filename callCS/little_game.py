import numpy as np

class Game:
    def __init__(self, size):
        self.size = size
        self.reset()


    def step(self):
        pass

    # noinspection PyAttributeOutsideInit
    def reset(self):
        #self.field = np.zeros((self.size, self.size))
        self.player = Player(self.size // 2, self.size // 2, 1)
        self.creatures = {self.player}

    def get_observations(self):
        pass


class Creature:
    def __init__(self, x, y, velocity):
        self.x = x
        self.y = y
        self.velocity = velocity

    def move(self, dx, dy):
        self.x += self.velocity * dx
        self.y += self.velocity * dy

    def intersects(self, creature: 'Creature'):
        return creature.y == self.y and creature.x == self.x


class Player(Creature):
    def shoot(self, dx, dy):
        pass


class Monster:
    pass

class Bullet:
    pass