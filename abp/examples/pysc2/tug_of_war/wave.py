class Wave():
    def __init__(self,  wave_data):

        self.economy = Economy()
        self.economy.self_mineral = wave_data[0]

        self.top = Lane()
        self.bottom = Lane()
        self.buildings_self_top          = self.top.buildings_self          = Buildings()
        self.buildings_self_top.marine   = self.top.buildings_self.marine   = wave_data[1]
        self.buildings_self_top.baneling = self.top.buildings_self.baneling = wave_data[2]
        self.buildings_self_top.immortal = self.top.buildings_self.immortal = wave_data[3]


        self.buildings_self_bottom          = self.bottom.buildings_self          = Buildings()
        self.buildings_self_bottom.marine   = self.bottom.buildings_self.marine   = wave_data[4]
        self.buildings_self_bottom.baneling = self.bottom.buildings_self.baneling = wave_data[5]
        self.buildings_self_bottom.immortal = self.bottom.buildings_self.immortal = wave_data[6]

        self.economy.self_pylon = wave_data[7]


        self.buildings_enemy_top          = self.top.buildings_enemy          = Buildings()
        self.buildings_enemy_top.marine   = self.top.buildings_enemy.marine   = wave_data[8]
        self.buildings_enemy_top.baneling = self.top.buildings_enemy.baneling = wave_data[9]
        self.buildings_enemy_top.immortal = self.top.buildings_enemy.immortal = wave_data[10]


        self.buildings_enemy_bottom          = self.bottom.buildings_enemy          = Buildings()
        self.buildings_enemy_bottom.marine   = self.bottom.buildings_enemy.marine   = wave_data[11]
        self.buildings_enemy_bottom.baneling = self.bottom.buildings_enemy.baneling = wave_data[12]
        self.buildings_enemy_bottom.immortal = self.bottom.buildings_enemy.immortal = wave_data[13]


        self.economy.enemy_pylon = wave_data[14]

        self.units_self_top          = self.top.units_self          = Units()
        self.units_self_top.marine   = self.top.units_self.marine   = wave_data[15]
        self.units_self_top.baneling = self.top.units_self.baneling = wave_data[16]
        self.units_self_top.immortal = self.top.units_self.immortal = wave_data[17]

        self.units_self_bottom          = self.bottom.units_self          = Units()
        self.units_self_bottom.marine   = self.bottom.units_self.marines  = wave_data[18]
        self.units_self_bottom.baneling = self.bottom.units_self.baneling = wave_data[19]
        self.units_self_bottom.immortal = self.bottom.units_self.immortal = wave_data[20]

        self.units_enemy_top          = self.top.units_enemy           =  Units()
        self.units_enemy_top.marine   = self.top.units_enemy.marine    = wave_data[21]
        self.units_enemy_top.baneling = self.top.units_enemy.bangeling = wave_data[22]
        self.units_enemy_top.immortal = self.top.units_enemy.immortal  = wave_data[23]

        self.units_enemy_bottom          = self.bottom.units_enemy              = Units()
        self.units_enemy_bottom.marine   = self.bottom.units_enemy.marine       = wave_data[24]
        self.units_enemy_bottom.baneling = self.bottom.units_enemy.bangeling    = wave_data[25]
        self.units_enemy_bottom.immortal = self.bottom.units_enemy.immortal     = wave_data[26]

        self.nexus_self_top     = self.top.nexus_self     = wave_data[27]
        self.nexus_self_bottom  = self.bottom.nexus_self  = wave_data[28]
        self.nexus_enemy_top    = self.top.nexus_enemy    = wave_data[29]
        self.nexus_enemy_bottom = self.bottom.nexus_enemy = wave_data[30]

class Lane():
    def __init__(self):
        self.buildings_self = 0
        self.buildings_enemy = 0
        self.units_self = 0
        self.units_enemy = 0
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
