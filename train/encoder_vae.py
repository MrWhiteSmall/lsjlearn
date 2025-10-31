def get_vae():
    from diffusers.models import AutoencoderKL
    vae_path = '/mnt/nas/shengjie/huggingface_model_local/sd-vae'
    vae = AutoencoderKL.from_pretrained(vae_path)
    vae.cuda()

    return vae

def get_vae_output(vae,x):
    import torch
    with torch.no_grad():
        latents = vae.encode(x).latent_dist.sample()
    return latents * 0.18215  # 缩放因子

def get_vae_decoded(vae,x):
    x /= 0.18215
    import torch
    with torch.no_grad():
        output = vae.decode(x).sample
    return output
if __name__=='__main__':
    import sys, os
    sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
    from utils.util_for_argparse import get_cuda_args
    get_cuda_args()
    vae = get_vae()

    import torch
    input = torch.randn((2,3,512,512)).cuda() # dict_keys(['sample', 'commit_loss'])
    output = get_vae_output(vae, input)
    print('encoded: ', output.shape)
    
    decoded = get_vae_decoded(vae,output)
    print('decoded: ', decoded.shape)
    
    pass