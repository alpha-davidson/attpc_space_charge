import sys
sys.path.append('..')
import torch
from models.common import *
from models.vae_gaussian import *
from models.vae_flow import *
import numpy as np
from torch.utils.data import DataLoader, TensorDataset
import argparse

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
# sim = SC
# exp = noSC

# Arguments
parser = argparse.ArgumentParser()
parser.add_argument('--sim_model', type=str, default='logs_gen/fission_sim_yesSC_100/ckpt_58400.000000_2277639.pt')
parser.add_argument('--exp_model', type=str, default='logs_gen/fission_sim_noSC_100/ckpt_57500.000000_2127537.pt')
parser.add_argument('--sim_data', type=str, default='data/Fission/fission_data/Fission_sim_yesSC_sampled_scaled_XYZC_100.npy')
parser.add_argument('--exp_data', type=str, default='data/Fission/fission_data/Fission_sim_noSC_sampled_scaled_XYZC_100.npy')
parser.add_argument('--batch_size', type=int, default=50)
parser.add_argument('--output_prefix', type=str, default='roundtrip')
args = parser.parse_args()

# Load the pretrained models
ckpt = torch.load(args.sim_model, map_location=torch.device(device))
model_sim = GaussianVAE(ckpt['args']).to(device)
model_sim.load_state_dict(ckpt['state_dict'])
model_sim.eval()

ckpt = torch.load(args.exp_model, map_location=torch.device(device))
model_exp = GaussianVAE(ckpt['args']).to(device)
model_exp.load_state_dict(ckpt['state_dict'])
model_exp.eval()

print("UPDATE: Models loaded.")

# Load the data
sim_data = np.load(args.sim_data)
exp_data = np.load(args.exp_data)

sim_data = torch.from_numpy(sim_data).float()
exp_data = torch.from_numpy(exp_data).float()

print("UPDATE: Data loaded.")
print("     Sim Data Shape:", sim_data.shape)
print("     Exp Data Shape:", exp_data.shape)

sim_dset = TensorDataset(sim_data)
exp_dset = TensorDataset(exp_data)

sim_loader = DataLoader(
    sim_dset,
    batch_size=args.batch_size,
    shuffle=False,
    num_workers=4,
    pin_memory=True
)

exp_loader = DataLoader(
    exp_dset,
    batch_size=args.batch_size,
    shuffle=False,
    num_workers=4,
    pin_memory=True
)


                #####################################
                ### Util functions for Cycle Diff ###
                #####################################

def forward_diffusion(x_0, model):
    diffusion = model.diffusion
    num_steps = diffusion.var_sched.num_steps
    trajectory = [x_0]
    for t in range(1, num_steps + 1):
        beta = diffusion.var_sched.betas[t]
        c0 = torch.sqrt(beta).view(-1, 1, 1)
        c1 = torch.sqrt(1 - beta).view(-1, 1, 1)
        e_rand = torch.randn_like(x_0)
        x_t = c1 * trajectory[-1] + c0 * e_rand
        trajectory.append(x_t)
    return trajectory


def dpm_encoder(x_0, model, context):
    with torch.no_grad():
        batch_size, num_point, dim = x_0.shape
        traj = forward_diffusion(x_0, model)
        diffusion = model.diffusion
        epsilon_list = []
        x_T = traj[-1]
        for t in range(len(traj)-1, 0, -1):
            alpha = diffusion.var_sched.alphas[t]
            alpha_bar = diffusion.var_sched.alpha_bars[t]
            sigma = diffusion.var_sched.get_sigmas(t, flexibility=0.1)
            c0 = 1.0 / torch.sqrt(alpha)
            c1 = (1 - alpha) / torch.sqrt(1 - alpha_bar)
            beta = diffusion.var_sched.betas[[t]*batch_size]
            e_theta = diffusion.net(traj[t], beta=beta, context=context)
            mean = c0 * (traj[t] - c1 * e_theta)
            epsilon = (traj[t-1] - mean) / sigma
            epsilon_list.append(epsilon)
        eps_tensor = torch.stack(epsilon_list, dim=1)
    return x_T, eps_tensor


def sample(model, num_points, context, x_T, eps, point_dim=4, flexibility=0.1, ret_traj=False):
    batch_size = context.size(0)
    diffusion = model.diffusion
    traj = {diffusion.var_sched.num_steps: x_T}
    for t in range(diffusion.var_sched.num_steps, 0, -1):
        z = eps[:, diffusion.var_sched.num_steps-t] if t > 1 else torch.zeros_like(x_T)
        alpha = diffusion.var_sched.alphas[t]
        alpha_bar = diffusion.var_sched.alpha_bars[t]
        sigma = diffusion.var_sched.get_sigmas(t, flexibility)
        c0 = 1.0 / torch.sqrt(alpha)
        c1 = (1 - alpha) / torch.sqrt(1 - alpha_bar)
        x_t = traj[t]
        beta = diffusion.var_sched.betas[[t]*batch_size]
        e_theta = diffusion.net(x_t, beta=beta, context=context)
        x_next = c0 * (x_t - c1 * e_theta) + sigma * z
        traj[t-1] = x_next.detach()
        traj[t] = traj[t].cpu()
        if not ret_traj:
            del traj[t]
    if ret_traj:
        return traj
    else:
        return traj[0]


def unscale_pc(pc_scaled):
    """[x/250, y/250, z/500+1] -> [x,y,z] in mm."""
    pts = pc_scaled.clone()
    pts[:, :, 0] = pts[:, :, 0] * 250.0
    pts[:, :, 1] = pts[:, :, 1] * 250.0
    pts[:, :, 2] = (pts[:, :, 2] - 1.0) * 500.0
    pts[:, :, 3] = torch.pow(10, pts[:, :, 3])
    return pts


                #####################################
                ## Round Trip: SC -> noSC -> SC    ##
                #####################################

all_orig = []
all_trans = []
all_recon = []

with torch.no_grad():
    for i, batch in enumerate(sim_loader):
        data = batch[0].to(device)

        # ── Forward: SC -> noSC ──
        mu_sim, sigma_sim = model_sim.encoder(data)
        context_sim = reparameterize_gaussian(mean=mu_sim, logvar=sigma_sim)

        mu_exp, sigma_exp = model_exp.encoder(data)
        context_exp = reparameterize_gaussian(mean=mu_exp, logvar=sigma_exp)

        x_T, eps = dpm_encoder(data, model_sim, context_sim)
        x_T = x_T.to(device)
        eps = eps.to(device)
        print(f"Batch {i}/{len(sim_loader)}: forward encoded")
        x = sample(model_exp, 512, context_exp, x_T, eps)

        # ── Reverse: noSC -> SC ──
        mu_exp2, sigma_exp2 = model_exp.encoder(x)
        context_exp2 = reparameterize_gaussian(mean=mu_exp2, logvar=sigma_exp2)

        mu_sim2, sigma_sim2 = model_sim.encoder(x)
        context_sim2 = reparameterize_gaussian(mean=mu_sim2, logvar=sigma_sim2)

        x_T2, eps2 = dpm_encoder(x, model_exp, context_exp2)
        x_T2 = x_T2.to(device)
        eps2 = eps2.to(device)
        print(f"Batch {i}/{len(sim_loader)}: reverse encoded")
        x2 = sample(model_sim, 512, context_sim2, x_T2, eps2)

        all_orig.append(unscale_pc(data).cpu())
        all_trans.append(unscale_pc(x).cpu())
        all_recon.append(unscale_pc(x2).cpu())

orig = torch.cat(all_orig, dim=0)
trans = torch.cat(all_trans, dim=0)
recon = torch.cat(all_recon, dim=0)

np.save(f'{args.output_prefix}_original.npy', orig.numpy())
np.save(f'{args.output_prefix}_translated.npy', trans.numpy())
np.save(f'{args.output_prefix}_reconstructed.npy', recon.numpy())
print(f"Saved: original {orig.shape}, translated {trans.shape}, reconstructed {recon.shape}")