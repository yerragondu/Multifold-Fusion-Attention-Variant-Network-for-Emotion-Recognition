from torch import nn

from scripts.models.v1 import MMFA

def generate_model(opt):
    assert opt.model in ['MMFA']

    if opt.model == 'MMFA':   
        model = MMFA.MultiModalCNN(opt.n_classes, fusion = opt.fusion, seq_length = opt.sample_duration, pretr_ef=opt.pretrain_path, num_heads=opt.num_heads)


    if opt.device != 'cpu':
        model = model.to(opt.device)
        model = nn.DataParallel(model, device_ids=None)
        pytorch_total_params = sum(p.numel() for p in model.parameters() if
                               p.requires_grad)
        print("Total number of trainable parameters: ", pytorch_total_params)
        
    
    return model, model.parameters()