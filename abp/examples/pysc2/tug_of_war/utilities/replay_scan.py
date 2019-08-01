import os
import sys
import json


unit_types = {
    21 : 'Mar_bldg', #'Barracks'
    28 : 'Ban_bldg', # 'Starport'
    70 : 'Imm_bldg', # 'RoboticsFacility'
    60 : 'Pylon', # 'Pylon'
    59 : 'Nexus', # 'Nexus'
    48 : 'M', # 'Marine'
    9 : 'B', # 'Baneling'
    83 : "I", # 'Immortal'
    45: "recorder_unit"
}

def load_json_from_replay_datafile(json_path):
    if not os.path.exists(json_path):
        print(f"ERROR: file not found: {json_path}")
        return None
    
    f = open(json_path)
    data = f.read()
    f.close()
    return data



if __name__ == "__main__":
    json_string = load_json_from_replay_datafile(sys.argv[1])
    if not (json_string == None):
        total_counts = {}
        pylon_appeared = False
        frames = json.loads(json_string)
        print (f"frame count : {len(frames)}")
        for frame in frames:
            frame_counts = {}
            for unit in frame["units"]:
                type = unit["unit_type"]
                if type == 60:
                    pylon_appeared = True
                    print(f'Pylon appeared in frame {frame["frame_number"]}')
        print(f"pylon unit found: {pylon_appeared}")
            #     if type != 45:
            #         note_unit(type, total_counts)
            #         note_unit(type, frame_counts)
            # print(f'frame {frame["frame_number"]} Mar_bldg')