def main():
    STEPS = 20000
    import abp.examples.scaii.city_attack.hand_policy as hand_policy
    from abp.examples.scaii.city_attack.bad import or_zero
    from scaii.env.sky_rts.env.scenarios.city_attack import CityAttack
    import operator
    import math

    env = CityAttack()

    reward_types = sorted(env.reward_types())

    state = env.reset()

    actions = env.actions()['actions']
    actions = sorted(actions.items(), key=operator.itemgetter(1))
    choice_descriptions = list(map(lambda x: x[0], actions))
    choices = list(map(lambda x: x[1], actions))

    hp_cum = {}

    for episode in range(STEPS):
        state = env.reset()
        total_reward = 0

        running_cum = {}
        for r_type in reward_types:
            running_cum[r_type] = []
        running_cum["total"] = []

        hps = []

        while not state.is_terminal():
            hps.append(tank_hp(state.objects))
            # print(hps)
            action = env.new_action()
            quad = hand_policy.eval_state(state)
            action.attack_quadrant(quad[0])

            state = env.act(action)

            total = 0
            for r_type in reward_types:
                reward = or_zero(state.typed_reward, r_type)
                for i, val in enumerate(running_cum[r_type]):
                    running_cum[r_type][i] += reward
                running_cum[r_type].append(reward)
                total += reward

            for i, val in enumerate(running_cum["total"]):
                running_cum["total"][i] += reward
            running_cum["total"].append(total)

        for i, hp in enumerate(hps):
            hp = int(round(hp))
            if hp not in hp_cum:
                hp_cum[hp] = {"total": 0}
                for r_type in reward_types:
                    hp_cum[hp][r_type] = 0

            for r_type in reward_types:
                hp_cum[hp][r_type] += running_cum[r_type][i]
            hp_cum[hp]["total"] += running_cum["total"][i]

        if episode % 100 == 0 and episode > 0:
            hp_avg = {}
            for k, m in hp_cum.items():
                hp_avg[k] = {}
                for r, v in m.items():
                    hp_avg[k][r] = v / episode

            print(hp_avg)
            return


def tank_hp(objects):
    from scaii.env.sky_rts.env.scenarios.city_attack import UnitType

    for _, obj in objects.items():
        # print(obj)
        if obj.unit_type == UnitType.TANK and obj.is_friendly:
            return obj.hp


if __name__ == "__main__":
    main()
