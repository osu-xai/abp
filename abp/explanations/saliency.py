import os

import torch
import torchvision
import numpy as np

# TODO: These modules do not yet exist in the repository
from saliency import SaliencyMethod, MapType, generate_saliency

use_cuda = torch.cuda.is_available()
FloatTensor = torch.cuda.FloatTensor if use_cuda else torch.FloatTensor
LongTensor = torch.cuda.LongTensor if use_cuda else torch.LongTensor
IntTensor = torch.cuda.IntTensor if use_cuda else torch.IntTensor
ByteTensor = torch.cuda.ByteTensor if use_cuda else torch.ByteTensor
Tensor = FloatTensor


class Saliency(object):
    """ Saliency for an Adaptive Variable """
    # TODO currenly supports only HRA Adaptive

    def __init__(self, adaptive):
        super(Saliency, self).__init__()
        self.adaptive = adaptive

    def generate_saliencies(self, step, state, choice_descriptions, layer_names,
                            file_path_prefix="saved_saliencies/", reshape=None):
        state = Tensor(state).unsqueeze(0)

        file_path_prefix = file_path_prefix + "step_" + str(step) + "/"
        methods = {}
        for saliency_method in SaliencyMethod:
            file_path = file_path_prefix + str(saliency_method) + "/"
            group_saliency = {}
            for idx, choice in enumerate(self.adaptive.choices):
                choice_saliency = {}
                self.adaptive.eval_model.model.combined = True
                #print('COMBINED '+str(choice_descriptions[idx])+' '+str(idx))
                saliencies = self.generate_saliency_for(
                    state, [idx], saliency_method)
                choice_saliency["all"] = saliencies[MapType.ABSOLUTE].detach(
                ).numpy().reshape((20,40,40))
                self.save_saliencies(saliencies, file_path + "choice_" +
                                     str(choice_descriptions[idx]
                                         ) + "/combined/",
                                     reshape, layer_names)
                self.adaptive.eval_model.model.combined = False
#                print('index: '+str(idx)+ ' choice: '+str(choice) + ' choice description: '+str(choice_descriptions[idx]))

                for reward_idx, reward_type in enumerate(self.adaptive.reward_types):
                    #print(str(reward_type)+' '+str(choice_descriptions[idx])+' '+str(reward_idx))
                    saliencies = self.generate_saliency_for(
                        state, [idx], saliency_method, reward_idx)
                    self.save_saliencies(saliencies, file_path + "choice_" + str(
                        choice_descriptions[idx]) + "/" + "reward_type_" + str(reward_type) + "/",
                        reshape, layer_names)
                    choice_saliency[reward_type] = saliencies[MapType.ABSOLUTE].detach(
                    ).numpy().reshape((20,40,40))
                group_saliency[choice] = choice_saliency
            methods[saliency_method] = group_saliency

        return methods[SaliencyMethod.PERTURBATION_2]

    def save_saliencies(self, saliencies, file_path_prefix, reshape, layer_names):
        for map_type, saliency in saliencies.items():
            # saliency = saliency.view(*reshape) #(40, 40, 8)
            # for idx, layer_name in enumerate(layer_names):
                #print('index: '+str(idx)+ ' layer: '+str(layer_name))
            if not os.path.exists(file_path_prefix + str(map_type)):
                os.makedirs(file_path_prefix + str(map_type))
            # print(saliency)
            # if str(map_type) == 'MapType.INPUT':  # input
            #saliency = saliency.view(*reshape)
            saliency = saliency.view(20, 40, 40)
            # for idx in range(17):
            #     torchvision.utils.save_image(
            #         (saliency[:, :, idx]),
            #         file_path_prefix +
            #         str(map_type) + "/"  + str(idx) +".png",
            #         normalize=True)
            for idx, layer_name in enumerate(layer_names):
                 torchvision.utils.save_image(
                     (saliency[idx, :, :]),
                     file_path_prefix +
                     str(map_type) + "/" + str(layer_name) + ".png",
                     normalize=True)
            # elif saliency.shape[1] == 8000:
            #     saliency = saliency.view(40, 40, 5)
            #     torchvision.utils.save_image(
            #         (saliency[:, :, 0].transpose(0, 1)),
            #         file_path_prefix + str(map_type) + "/" + "HP" + ".png",
            #         normalize=False)
            #     torchvision.utils.save_image(
            #         (saliency[:, :, 1].transpose(0, 1)),
            #         file_path_prefix + str(map_type) +
            #         "/" + "Enemy Tank" + ".png",
            #         normalize=False)
            #     torchvision.utils.save_image(
            #         (saliency[:, :, 2].transpose(0, 1)),
            #         file_path_prefix + str(map_type) + "/" + "Size" + ".png",
            #         normalize=False)
            #     torchvision.utils.save_image(
            #         (saliency[:, :, 3].transpose(0, 1)),
            #         file_path_prefix + str(map_type) + "/" + "Type" + ".png",
            #         normalize=False)
            #     torchvision.utils.save_image(
            #         (saliency[:, :, 4].transpose(0, 1)),
            #         file_path_prefix + str(map_type) +
            #         "/" + "Friend or Enemy" + ".png",
            #         normalize=False)
            # else:
            #     saliency = saliency.view(*reshape)
            #     torchvision.utils.save_image(
            #         (saliency[:, :, 0].transpose(0, 1)),
            #         file_path_prefix + str(map_type) + "/" + "HP" + ".png",
            #         normalize=False)
            #     torchvision.utils.save_image(
            #         saliency[:, :, 1].transpose(0, 1),
            #         file_path_prefix + str(map_type) + "/" + "Tank" + ".png",
            #         normalize=False)
            #     # torchvision.utils.save_image(
            #     #                     (saliency[:, :, 2] + saliency[:, :, 3]).transpose(0, 1)/2,
            #     #                     file_path_prefix + str(map_type) + "/" + "Small or Big Towers" + ".png",
            #     #                     normalize=True)
            #     # torchvision.utils.save_image(
            #     #                     ((saliency[:, :, 4] + saliency[:, :, 5]).transpose(0, 1))/2,
            #     #                     file_path_prefix + str(map_type) + "/" + "Small or Big Cities" + ".png",
            #     #                     normalize=True)
            #     torchvision.utils.save_image(
            #         (saliency[:, :, 2] + saliency[:, :, 3] + saliency[:,
            #                                                           :, 4] + saliency[:, :, 5]).transpose(0, 1) / 4,
            #         file_path_prefix + str(map_type) + "/" +
            #         "Small or Big Towers or Cities" + ".png",
            #         normalize=False)
            #     torchvision.utils.save_image(
            #         ((saliency[:, :, 6] + saliency[:, :, 7]
            #           ).transpose(0, 1)) / 2,
            #         file_path_prefix + str(map_type) +
            #         "/" + "Friend or Enemy" + ".png",
            #         normalize=False)

    def generate_saliency_for(self, state, choice, saliency_method, reward_idx=None):
        model = self.adaptive.eval_model.model

        if reward_idx is not None:
            model = self.adaptive.eval_model.get_model_for(reward_idx)

        if type(choice) == int:
            choice = [choice]

        return generate_saliency(model, state, choice, type=saliency_method)
