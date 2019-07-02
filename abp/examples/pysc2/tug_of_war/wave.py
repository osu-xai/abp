class Wave():
    def __init__(self,  wave_data):

        self.economy = Economy()
        self.economy.self_mineral = wave_data[0]

        self.building_self_top = Buildings()
        self.building_self_top.marine = wave_data[1]
        self.building_self_top.baneling = wave_data[2]
        self.building_self_top.immortal = wave_data[3]


        self.building_self_bottom = Buildings()
        self.building_self_bottom.marine = wave_data[4]
        self.building_self_bottom.baneling = wave_data[5]
        self.building_self_bottom.immortal = wave_data[6]

        self.economy.self_pylon = wave_data[7]


        self.building_enemy_top = Buildings()
        self.building_enemy_top.marine = wave_data[8]
        self.building_enemy_top.baneling = wave_data[9]
        self.building_enemy_top.immortal = wave_data[10]


        self.building_enemy_bottom = Buildings()
        self.building_enemy_bottom.marine = wave_data[11]
        self.building_enemy_bottom.baneling = wave_data[12]
        self.building_enemy_bottom.immortal = wave_data[13]


        self.economy.enemy_pylon = wave_data[14]

        self.unit_self_top = Units()
        self.unit_self_top.marine = wave_data[15]
        self.unit_self_top.baneling = wave_data[16]
        self.unit_self_top.immortal = wave_data[17]

        self.unit_self_bottom = Units()
        self.unit_self_bottom.marine = wave_data[18]
        self.unit_self_bottom.baneling = wave_data[19]
        self.unit_self_bottom.immortal = wave_data[20]

        self.unit_enemy_top = Units()
        self.unit_enemy_top.marine = wave_data[21]
        self.unit_enemy_top.baneling = wave_data[22]
        self.unit_enemy_top.immortal = wave_data[23]

        self.unit_enemy_bottom = Units()
        self.unit_enemy_bottom.marine = wave_data[24]
        self.unit_enemy_bottom.baneling = wave_data[25]
        self.unit_enemy_bottom.immortal = wave_data[26]

        self.nexus_self_top = wave_data[27]
        self.nexus_self_bottom = wave_data[28]
        self.nexus_enemy_top = wave_data[29]
        self.nexus_enemy_bottom = wave_data[30]

class Lane():
    def __init__(self):
        self.building_self = 0
        self.building_enemy = 0
        self.unit_self = 0
        self.unit_enemy = 0
        self.nexus_self = 0
        self.nexus_enemy = 0

class Economy():
    def __init__(self):
        self.self_mineral = 0
        self.self_pylon = 0
        self.enemy_pylon = 0


class Buildings():
    def __init__(self):
        self.marine = 0
        self.baneling = 0
        self.immortal = 0


class Units():
    def __init__(self):
        self.marine = 0
        self.baneling = 0
        self.immortal = 0