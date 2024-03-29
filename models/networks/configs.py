# Copyright (c) Facebook, Inc. and its affiliates. All rights reserved.

def get_resnet_arch(model_type, opt, in_channels=3):
    setup = model_type.split("_")[1]

    if setup == "256W8UpDown":
        arch = {
            "layers_enc": [
                in_channels,
                opt.ngf // 2,
                opt.ngf // 2,
                opt.ngf // 2,
                opt.ngf,
                opt.ngf,
                opt.ngf,
                opt.ngf,
                64,
            ],
            "downsample": [
                False,
                False,
                False,
                False,
                False,
                False,
                False,
                False,
            ],
            "layers_dec": [
                128,
                opt.ngf,
                opt.ngf * 2,
                opt.ngf * 4,
                opt.ngf * 4,
                opt.ngf * 2,
                opt.ngf * 2,
                opt.ngf * 2,
                3,
            ],
            "upsample": [
                False,
                "Down",
                "Down",
                False,
                "Up",
                "Up",
                False,
                False,
            ],
            "non_local": False,
            "non_local_index": 1,
        }

    elif setup == "256W8UpDown64":
        if opt.concat == True:
            first_layer=67
        else:
            first_layer=64
        arch = {
            "layers_enc": [
                in_channels,
                opt.ngf // 2,
                opt.ngf // 2,
                opt.ngf // 2,
                opt.ngf,
                opt.ngf,
                opt.ngf,
                opt.ngf,
                64,
            ],
            "downsample": [
                False,
                False,
                False,
                False,
                False,
                False,
                False,
                False,
            ],
            "layers_dec": [
                first_layer,
                opt.ngf,
                opt.ngf * 2,
                opt.ngf * 4,
                opt.ngf * 4,
                opt.ngf * 2,
                opt.ngf * 2,
                opt.ngf * 2,
                3,
            ],
            "upsample": [
                False,
                "Down",
                "Down",
                False,
                "Up",
                "Up",
                False,
                False,
            ],
            "non_local": False,
            "non_local_index": 1,
        }

    elif setup == "256W8UpDownDV":
        arch = {
            "layers_enc": [
                in_channels,
                opt.ngf // 2,
                opt.ngf // 2,
                opt.ngf // 2,
                opt.ngf,
                opt.ngf,
                opt.ngf,
                opt.ngf,
                64,
            ],
            "downsample": [
                False,
                False,
                False,
                False,
                False,
                False,
                False,
                False,
            ],
            "layers_dec": [
                64,
                opt.ngf,
                opt.ngf * 2,
                opt.ngf * 4,
                opt.ngf * 4,
                opt.ngf * 2,
                opt.ngf * 2,
                opt.ngf * 2,
                3,
            ],
            "upsample": [
                False,
                "Down",
                "Down",
                False,
                "Up",
                "Up",
                False,
                False,
            ],
            "non_local": False,
            "non_local_index": 1,
        }

    elif setup == "256W8UpDownRGB":
        arch = {
            "layers_enc": [
                in_channels,
                opt.ngf // 2,
                opt.ngf // 2,
                opt.ngf // 2,
                opt.ngf,
                opt.ngf,
                opt.ngf,
                opt.ngf,
                64,
            ],
            "downsample": [
                False,
                False,
                False,
                False,
                False,
                False,
                False,
                False,
            ],
            "layers_dec": [
                6,
                opt.ngf,
                opt.ngf * 2,
                opt.ngf * 4,
                opt.ngf * 4,
                opt.ngf * 2,
                opt.ngf * 2,
                opt.ngf * 2,
                3,
            ],
            "upsample": [
                False,
                "Down",
                "Down",
                False,
                "Up",
                "Up",
                False,
                False,
            ],
            "non_local": False,
            "non_local_index": 1,
        }

    elif setup == "256W8UpDown3":
        arch = {
            "layers_enc": [
                in_channels,
                opt.ngf // 2,
                opt.ngf // 2,
                opt.ngf // 2,
                opt.ngf,
                opt.ngf,
                opt.ngf,
                opt.ngf,
                64,
            ],
            "downsample": [
                False,
                False,
                False,
                False,
                False,
                False,
                False,
                False,
            ],
            "layers_dec": [
                3,
                opt.ngf,
                opt.ngf * 2,
                opt.ngf * 4,
                opt.ngf * 4,
                opt.ngf * 2,
                opt.ngf * 2,
                opt.ngf * 2,
                3,
            ],
            "upsample": [
                False,
                "Down",
                "Down",
                False,
                "Up",
                "Up",
                False,
                False,
            ],
            "non_local": False,
            "non_local_index": 1,
        }

    elif setup == "256W8":
        arch = {
            "layers_enc": [
                in_channels,
                opt.ngf,
                opt.ngf,
                opt.ngf * 2,
                opt.ngf * 2,
                opt.ngf * 2,
                opt.ngf * 4,
                opt.ngf * 4,
                64,
            ],
            "downsample": [
                True,
                False,
                False,
                False,
                True,
                False,
                False,
                False,
            ],
            "layers_dec": [
                64,
                opt.ngf,
                opt.ngf,
                opt.ngf * 2,
                opt.ngf * 2,
                opt.ngf * 2,
                opt.ngf * 4,
                opt.ngf * 4,
                3,
            ],
            "upsample": [False, False, True, False, False, False, True, False],
            "non_local": False,
            "non_local_index": 1,
        }

    return arch
