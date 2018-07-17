from functools import total_ordering
TANK_DAMAGE = 10
BIG_FORT_DAMAGE = 10
SMALL_FORT_DAMAGE = 5

BIG_FORT_REWARD = 70
SMALL_FORT_REWARD = 50

BIG_CITY_PENALTY = 150
SMALL_CITY_PENALTY = 120


def eval_state(state):
    from scaii.env.sky_rts.env.scenarios.city_attack import UnitType, Actions

    enemy_tank = None
    tank = None

    quadrants = dict([])
    tank_quadrant = None
    for _, obj in state.objects.items():
        if obj.unit_type == UnitType.TANK:
            if not obj.is_friendly:
                enemy_tank = obj
            else:
                tank = obj
            continue

        centroid_x = (obj.min_x + obj.max_x) / 2
        centroid_y = (obj.min_y + obj.max_y) / 2

        quadrants[find_quadrant(centroid_x, centroid_y)] = obj

    if not enemy_tank == None:
        centroid_x = (enemy_tank.min_x + enemy_tank.max_x) / 2
        centroid_y = (enemy_tank.min_y + enemy_tank.max_y) / 2

        tank_quadrant = find_quadrant(centroid_x, centroid_y)

    objects = [ComparableUnit(unit, tank, quadrant, enemy_tank if tank_quadrant ==
                              quadrant else None) for quadrant, unit in quadrants.items()]
    objects.sort()

    return (objects[-1].quadrant, objects)


def find_quadrant(centroid_x, centroid_y):
    from scaii.env.sky_rts.env.scenarios.city_attack import Actions
    CENTER_X, CENTER_Y = 20, 20

    #print(centroid_x, centroid_y)

    if centroid_x > CENTER_X and centroid_y < CENTER_Y:
        return Actions.Q1
    elif centroid_x < CENTER_X and centroid_y < CENTER_Y:
        return Actions.Q2
    elif centroid_x < CENTER_X and centroid_y > CENTER_Y:
        return Actions.Q3
    elif centroid_x > CENTER_X and centroid_y > CENTER_Y:
        return Actions.Q4
    else:
        return "tank"


@total_ordering
class ComparableUnit():
    def __init__(self, unit, tank, quadrant, enemy_tank):
        self.unit = unit
        # hack
        self.tank = tank
        self.quadrant = quadrant
        self.enemy_tank = enemy_tank
        self.hp = self.unit.hp
        self.is_friendly = self.unit.is_friendly
        self.unit_type = self.unit.unit_type

    def will_kill_us(self):
        return will_kill_us(self.unit, self.tank)

    def will_be_killed(self):
        return not self.will_kill_us()

    def __eq__(self, other):
        return self.unit.min_x == other.unit.min_x and self.unit.min_y == other.unit.min_y

    def __gt__(self, other):
        tank = self.tank

        if self.enemy_tank != None and other.is_friendly:
            return True
        elif self.is_friendly and other.enemy_tank != None:
            return False
        elif self.enemy_tank != None and (not other.is_friendly):
            return other.will_kill_us() or city_will_die(self, self.enemy_tank, other)
        elif (not self.is_friendly) and other.enemy_tank != None:
            return not (self.will_kill_us() or city_will_die(other, other.enemy_tank, self))

        # Friendly always is bad
        if self.is_friendly != other.is_friendly:
            return not self.is_friendly
        elif self.is_friendly and other.is_friendly:
            return immediate_reward(self.unit, self.enemy_tank, self.tank) >= immediate_reward(other.unit, other.enemy_tank, other.tank)

        # Cases where we might die
        if self.will_kill_us() != other.will_kill_us():
            return self.will_be_killed()
        elif self.will_kill_us() and other.will_kill_us():
            return immediate_reward(self.unit, self.enemy_tank, self.tank) >= immediate_reward(other.unit, other.enemy_tank, other.tank)
        else:  # i.e. we could kill both
            if self.unit_type != other.unit_type:
                return tower_pref(self.unit, other.unit)
            else:
                return self.hp < other.hp


def tower_pref(fort, other):
    from scaii.env.sky_rts.env.scenarios.city_attack import UnitType

    pref_factor = BIG_FORT_DAMAGE / SMALL_FORT_DAMAGE

    if fort.unit_type == UnitType.BIG_FORT:
        return ticks_til_death(fort.hp, TANK_DAMAGE) < ticks_til_death(other.hp, TANK_DAMAGE) * pref_factor
    else:
        return ticks_til_death(fort.hp, TANK_DAMAGE) * pref_factor > ticks_til_death(other.hp, TANK_DAMAGE)


def city_will_die(city, enemy_tank, fort):
    city_dies_in = ticks_til_death(city.hp, TANK_DAMAGE)
    fort_dies_in = ticks_til_death(fort.hp, TANK_DAMAGE) + 3

    return city_dies_in < fort_dies_in


def will_kill_us(unit, tank):
    return next_hp(unit, tank) <= 0


def immediate_reward(unit, enemy_tank, tank):
    from scaii.env.sky_rts.env.scenarios.city_attack import UnitType
    if enemy_tank != None:
        enemy_kill_ticks = ticks_til_death(enemy_tank.hp, TANK_DAMAGE)
        # approximation of the extra negative reward we'll get while
        # travelling to the tank
        negative_ticks = enemy_kill_ticks + 2

        return negative_ticks * -TANK_DAMAGE

    if unit.is_friendly:
        kill_ticks = ticks_til_death(unit.hp, TANK_DAMAGE)

        penalty = None
        if unit.unit_type == UnitType.BIG_FORT:
            # print("foo")
            penalty = BIG_FORT_REWARD
        elif unit.unit_type == UnitType.SMALL_FORT:
            # print("b")
            penalty = SMALL_FORT_REWARD
        elif unit.unit_type == UnitType.BIG_CITY:
            # print("c")
            penalty = BIG_CITY_PENALTY
        elif unit.unit_type == UnitType.SMALL_CITY:
            # print("d")
            penalty = SMALL_CITY_PENALTY

        # print(unit.unit_type)

        return (kill_ticks * -TANK_DAMAGE) - penalty

    bonus = None
    damage_factor = None

    if unit.unit_type == UnitType.BIG_FORT:
        bonus = BIG_FORT_REWARD
        damage_factor = BIG_FORT_DAMAGE
    elif unit.unit_type == UnitType.SMALL_FORT:
        bonus = SMALL_FORT_REWARD
        damage_factor = SMALL_FORT_DAMAGE

    #print(damage_factor, unit)
    we_die_in = ticks_til_death(tank.hp, damage_factor)
    they_die_in = ticks_til_death(unit.hp, TANK_DAMAGE)

    # -1 to estimate us getting hit an extra time due to travel
    if we_die_in - 1 < they_die_in:
        we_die_in = we_die_in - 1
        return we_die_in * TANK_DAMAGE
    else:
        return (they_die_in * TANK_DAMAGE) + bonus


def ticks_til_death(hp, dmg_per_tick):
    import math
    assert(hp != None and dmg_per_tick != None)
    return math.ceil(hp / dmg_per_tick)


def _next_hp_raw(unit, tank):
    import math
    from scaii.env.sky_rts.env.scenarios.city_attack import UnitType
    if unit.is_friendly:
        return tank.hp

    damage_factor = None
    if unit.unit_type == UnitType.BIG_FORT:
        damage_factor = BIG_FORT_DAMAGE
    elif unit.unit_type == UnitType.SMALL_FORT:
        damage_factor = SMALL_FORT_DAMAGE

    # number of ticks it us to kill a unit
    damage_ticks = ticks_til_death(unit.hp, TANK_DAMAGE)
    # Approximation of "extra damage" we take
    # approaching the tower or due to attack speed
    damage_ticks += 1

    return tank.hp - (damage_ticks * damage_factor)


def next_hp(unit, tank):
    return max(0, _next_hp_raw(unit, tank))
