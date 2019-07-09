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

    def sum_buildings(self):
        total_enemy_top = self.buildings_enemy_top.marine + self.buildings_enemy_top.baneling + self.buildings_enemy_top.immortal 
        total_enemy_bottom = self.buildings_enemy_bottom.marine + self.buildings_enemy_bottom.baneling + self.buildings_enemy_bottom.immortal
        
        total_self_top = self.buildings_self_top.marine + self.buildings_self_top.baneling + self.buildings_self_top.immortal 
        total_self_bottom = self.buildings_self_bottom.marine + self.buildings_self_bottom.baneling + self.buildings_self_bottom.immortal

        total = total_enemy_top + total_enemy_bottom + total_self_top + total_self_bottom
        return total


    def get_p1_building_totals(self):
        total_self_top = self.buildings_self_top.marine + self.buildings_self_top.baneling + self.buildings_self_top.immortal 
        total_self_bottom = self.buildings_self_bottom.marine + self.buildings_self_bottom.baneling + self.buildings_self_bottom.immortal
        return total_self_bottom + total_self_top

    def get_p2_building_totals(self):
        total_enemy_top = self.buildings_enemy_top.marine + self.buildings_enemy_top.baneling + self.buildings_enemy_top.immortal 
        total_enemy_bottom = self.buildings_enemy_bottom.marine + self.buildings_enemy_bottom.baneling + self.buildings_enemy_bottom.immortal
        return total_enemy_bottom + total_enemy_top

    def is_reset(self, prev_wave, curr_wave):
        if (prev_wave.sum_buildings() > curr_wave.sum_buildings()):
            return True
        else:
            return False
    


    def is_p1_win(self):
        t1 = self.top.nexus_self
        t2 = self.top.nexus_enemy
        b1 = self.bottom.nexus_self
        b2 = self.bottom.nexus_enemy
        lowest_p1 = -1
        next_lowest_p1 = -1
        equal_p1 = -1
        lowest_p2 = -1
        next_lowest_p2 = -1
        equal_p2 = -1
        if t1 < b1:
            lowest_p1 = t1
            next_lowest_p1 = b1
        elif b1 < t1:
            lowest_p1 = b1
            next_lowest_p1 = t1
        else:
            equal_p1 = b1

        if t2 < b2:
            lowest_p2 = t2
            next_lowest_p2 = b2
        elif b2 < t2:
            lowest_p2 = b2
            next_lowest_p2 = t2
        else:
            equal_p2 = b2

        if equal_p1 == -1 and equal_p2 == -1:
            if lowest_p1 > lowest_p2:
                return True
            elif lowest_p2 > lowest_p1:
                return False
            else:
                if next_lowest_p1 > next_lowest_p2:
                    return True
                elif next_lowest_p2 > next_lowest_p1:
                    return False
                else:
                    return False
        elif equal_p1 == -1 and equal_p2 != -1:
            if lowest_p1 > equal_p2:
                return True
            elif equal_p2 > lowest_p1:
                return False
            else:
                if next_lowest_p1 > equal_p2:
                    return True
                elif equal_p2 > next_lowest_p1:
                    return False
                else:
                    return False
        elif equal_p1 != -1 and equal_p2 == -1:
            if equal_p1 > lowest_p2:
                return True
            elif lowest_p2 > equal_p1:
                return False
            else:
                if equal_p1 > next_lowest_p2:
                    return True
                elif next_lowest_p2 > equal_p1:
                    return False
                else:
                    return False
        else:
            if equal_p1 > equal_p2:
                return True
            elif equal_p2 > equal_p1:
                return False 
            else:
                return False 



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
