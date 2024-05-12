
import torch

from vpr_models import sfrs, salad, anyloc, convap, mixvpr, netvlad


def get_model(method, backbone=None, descriptors_dimension=None, args=None, **kwargs):
    if method == "sfrs":
        model = sfrs.SFRSModel()
    elif method == "netvlad":
        model = netvlad.NetVLAD(descriptors_dimension=descriptors_dimension)
    elif method == "cosplace":
        model = torch.hub.load("gmberton/cosplace", "get_trained_model",
                               backbone=backbone, fc_output_dim=descriptors_dimension)
    elif method == "mixvpr":
        model = mixvpr.get_mixvpr(descriptors_dimension=descriptors_dimension)
    elif method == "convap":
        model = convap.get_convap(descriptors_dimension=descriptors_dimension)
    elif method == "eigenplaces":
        model = torch.hub.load("gmberton/eigenplaces", "get_trained_model",
                               backbone=backbone, fc_output_dim=descriptors_dimension)
    elif method == "anyloc":
        model = anyloc.AnyLocWrapper()
    elif method == "salad":
        model = salad.SaladWrapper()
    elif method == "deep_vpr_model":
        model, deep_vpr_model_config = torch.hub.load('../deep-visual-geo-localization-benchmark', 
                       'deep_vpr_model', 
                       source='local',
                    #    source='github',
                    **kwargs
                       )
        model = anyloc.AnyLocWrapper(model=model, resize=[224, 224])
        import json
        deep_vpr_model_config_str = json.dumps(deep_vpr_model_config, indent=4)
        with open(f"{args.output_folder}/deep_vpr_model_config.json","w") as f:
            f.write(deep_vpr_model_config_str)
    return model

