import torch
from models.common import *
from models.vae_gaussian import *
from models.vae_flow import *
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from torch.utils.data import DataLoader, TensorDataset
from sklearn.linear_model import RANSACRegressor, LinearRegression
from evaluation.evaluation_metrics import EMD_CD
import argparse

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
# sim = SC
# exp = noSC

# Arguments
parser = argparse.ArgumentParser()

# Model arguments
parser.add_argument('--sim_model', type=str, default='logs_gen/fission_sim_SC_epoch_1000/ckpt_5000.000000_315063.pt')
parser.add_argument('--exp_model', type=str, default='logs_gen/fission_sim_noSC_epoch_1000/ckpt_57500.000000_1840032.pt')
parser.add_argument('--sim_data', type=str, default='data/Fission/fission_data/data/Fission_sim_yesSC_sampled_scaled_XYZC.npy')
parser.add_argument('--exp_data', type=str, default='data/Fission/fission_data/data/Fission_sim_noSC_sampled_scaled_XYZC.npy')
parser.add_argument('--trans_ratio', type=float, default=1.0)

args = parser.parse_args()

# Load the pretrained models
ckpt = torch.load(args.sim_model, map_location=torch.device(device))
model_sim = GaussianVAE(ckpt['args']).to(device)
model_sim.load_state_dict(ckpt['state_dict'])
model_sim.eval();

ckpt = torch.load(args.exp_model, map_location=torch.device(device))
model_exp = GaussianVAE(ckpt['args']).to(device)
model_exp.load_state_dict(ckpt['state_dict'])
model_exp.eval();

print("UPDATE: Models loaded.")

# Load the datasets for translation

# Load the data
sim_data = np.load(args.sim_data)
exp_data = np.load(args.exp_data)

# Convert to PyTorch Tensors
sim_data = torch.from_numpy(sim_data).float()
exp_data = torch.from_numpy(exp_data).float()

print("UPDATE: Data loaded.")
print("     Sim Data Shape:", sim_data.shape)
print("     Exp Data Shape:", exp_data.shape)

# Create TensorDatasets
sim_dset = TensorDataset(sim_data)
exp_dset = TensorDataset(exp_data)

b_size = 50

exp_loader = DataLoader(
    exp_dset,
    batch_size=b_size,
    shuffle=False,
    num_workers=4,
    pin_memory=True
)
# exp_loader = DataLoader(
#     exp_dset,
#     batch_size=50,
#     num_workers=0,
# )

sim_loader = DataLoader(
    sim_dset,
    batch_size=b_size,
    shuffle=False,
    num_workers=4,
    pin_memory=True
)
# sim_loader = DataLoader(
#     sim_dset,
#     batch_size=50,
#     num_workers=0,
# )

                #####################################
                ### Util functions for Cycle Diff ###
                #####################################

def forward_diffusion(x_0, model):
    """
    Simulates the forward diffusion process in a diffusion model.

    Parameters:
        x_0 (torch.Tensor): The initial input tensor representing the starting point cloud.
        model (object): The diffusion model containing the variance schedule and other parameters.

    Returns:
        list: A list of tensors representing the trajectory of the forward diffusion process. 
              Each tensor corresponds to an intermediate noisy point cloud at a given timestep.
    """
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

def visualize_point_cloud(point_cloud, sample_index=0):
    """
    Visualize point cloud with charges

    Parameters:
        point_cloud (torch.Tensor): The initial input batch of tensors representing point clouds.
        sample_index: the index of the point cloud to visualize in the batch
    """
    # Extract the sample
    sample = point_cloud[sample_index]

    # Extract x, y, z coordinates
    x = sample[:, 0].numpy()
    y = sample[:, 1].numpy()
    z = sample[:, 2].numpy()
    c = sample[:, 3].numpy()

    # Create a 3D plot
    fig = plt.figure(figsize=(6,4))
    ax = fig.add_subplot(111, projection='3d')

    # Scatter plot
    ax.scatter(x, y, z, c = c, s=2, cmap=plt.cool())

    # Setting labels
    ax.set_xlabel('X axis')
    ax.set_ylabel('Y axis')
    ax.set_zlabel('Z axis')

    # Show the plot
    plt.show()
    
    
def dpm_encoder(x_0, model, context):
    """
    Implementation of the DPM_Encoder.

    This function generates a noisy trajectory of point clouds from an initial point cloud `x_0` and
    then encodes it using a diffusion model by computing the corresponding noise vectors for each
    step in the diffusion process. The function returns the final noisy point cloud and a tensor
    of noise vectors.

    Parameters:
        x_0 (torch.Tensor): The initial input tensor representing the starting point cloud.
        model (object): The diffusion model containing the network and variance schedule.
        context (torch.Tensor): the latent from pointnet encoder of the DPM model.

    Returns:
        tuple: A tuple containing:
            - x_T (torch.Tensor): The final noisy point cloud after the forward diffusion process.
            - eps_tensor (torch.Tensor): A tensor containing the noise vectors for each step in the diffusion process.
    """
    with torch.no_grad():
        batch_size, num_point, dim = x_0.shape
        traj = forward_diffusion(x_0, model)
        diffusion = model.diffusion
        
        epsilon_list = []
        x_T = traj[-1]  # The final noisy point cloud
        for t in range(len(traj)-1, 0, -1):
            
            alpha = diffusion.var_sched.alphas[t]
            alpha_bar = diffusion.var_sched.alpha_bars[t]
            sigma = diffusion.var_sched.get_sigmas(t, flexibility = 0.1)
            
            c0 = 1.0 / torch.sqrt(alpha)
            c1 = (1 - alpha) / torch.sqrt(1 - alpha_bar)
            
            beta = diffusion.var_sched.betas[[t]*batch_size]
            e_theta = diffusion.net(traj[t], beta=beta, context=context)
            mean = c0 * (traj[t] - c1 * e_theta)
            epsilon = (traj[t-1] - mean) / sigma
            epsilon_list.append(epsilon)
        # Convert the list of noise tensors to a single tensor
        eps_tensor = torch.stack(epsilon_list, dim=1)
    return x_T, eps_tensor 

def sample(model, num_points, context, x_T, eps, point_dim=4, flexibility=0.1, ret_traj=False):
        """
        Sample point cloud with the DPM.

        Parameters:
            model (object): The diffusion model containing the network and variance schedule.
            num_points (int): The number of points in the final point cloud.
            context (torch.Tensor): The context tensor for conditioning the model.
            x_T (torch.Tensor): The initial noisy point cloud.
            eps (torch.Tensor): A tensor containing the noise vectors for each step in the reverse diffusion process.
            point_dim (int): The dimensionality of each point in the point cloud (default is 4).
            flexibility (float): The flexibility parameter for adjusting the diffusion schedule (default is 0.1).
            ret_traj (bool): Whether to return the trajectory of intermediate steps (default is False).

        Returns:
            Union[dict, torch.Tensor]: The trajectory of intermediate steps as a dictionary if `ret_traj` is True, 
                                    or the final sampled point cloud as a tensor if `ret_traj` is False.
        """
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
            # print(x_t.shape)
            beta = diffusion.var_sched.betas[[t]*batch_size]
            e_theta = diffusion.net(x_t, beta=beta, context=context)
            x_next = c0 * (x_t - c1 * e_theta) + sigma * z
            # print("NEXT: ", x_next.shape)
            traj[t-1] = x_next.detach()     # Stop gradient and save trajectory.
            traj[t] = traj[t].cpu()         # Move previous output to CPU memory.
            if not ret_traj:
                del traj[t]
        
        if ret_traj:
            return traj
        else:
            return traj[0]
        

def visualize_point_clouds_side_by_side(point_cloud1, point_cloud2, index, sample_index=0):
    """
    Visualize point cloud with charges for two point clouds side by side

    Parameters:
        point_cloud1, point_cloud2, (torch.Tensor): The initial input batch of tensors representing point clouds.
        sample_index: the index of the point cloud to visualize in the batch
    """
    # Extract the samples
    sample1 = point_cloud1[sample_index]
    sample2 = point_cloud2[sample_index]

    # Extract x, y, z coordinates for both samples
    x1, y1, z1, c1 = sample1[:, 0], sample1[:, 1], sample1[:, 2], sample1[:, 3]
    x2, y2, z2, c2 = sample2[:, 0], sample2[:, 1], sample2[:, 2], sample2[:, 3]
    
    # Create a figure and a set of subplots
    fig, axs = plt.subplots(1, 2, figsize=(10, 5), subplot_kw={'projection': '3d'})
    
    # Scatter plot for the first point cloud
    pc1 = axs[0].scatter(x1, z1, y1, c=c1, s=2, cmap='cool', vmin=0, vmax=8000)
    axs[0].set_title('Original', weight='bold')
    axs[0].set_xlabel('X', weight='bold')
    axs[0].set_ylabel('Z', weight='bold')
    axs[0].set_zlabel('Y', weight='bold')
    axs[0].set_xlim([-250, 250])
    axs[0].set_ylim([0, 1000])
    axs[0].set_zlim([-250, 250])
    axs[0].set_box_aspect([1, 2, 1])
    # axs[0].set_xticks([-1, -0.5, 0, 0.5, 1])
    # axs[0].set_yticks([1, 1.5, 2, 2.5, 3])
    # axs[0].set_zticks([-1, -0.5, 0, 0.5, 1])
    # axs[0].set_xticklabels([-1, -0.5, 0, 0.5, 1], fontweight='bold')
    # axs[0].set_yticklabels([1, 1.5, 2, 2.5, 3], fontweight='bold')
    # axs[0].set_zticklabels([-1, -0.5, 0, 0.5, 1], fontweight='bold')
    
    # Scatter plot for the second point cloud
    pc2 = axs[1].scatter(x2, z2, y2, c=c2, s=2, cmap='cool', vmin=0, vmax=8000)
    axs[1].set_title('Translation', weight='bold')
    axs[1].set_xlabel('X', weight='bold')
    axs[1].set_ylabel('Z', weight='bold')
    axs[1].set_zlabel('Y', weight='bold')
    axs[1].set_xlim([-250, 250])
    axs[1].set_ylim([0, 1000])
    axs[1].set_zlim([-250, 250])
    axs[1].set_box_aspect([1, 2, 1])
    # axs[1].set_xticks([-1, -0.5, 0, 0.5, 1])
    # axs[1].set_yticks([1, 1.5, 2, 2.5, 3])
    # axs[1].set_zticks([-1, -0.5, 0, 0.5, 1])
    # axs[1].set_xticklabels([-1, -0.5, 0, 0.5, 1], fontweight='bold')
    # axs[1].set_yticklabels([1, 1.5, 2, 2.5, 3], fontweight='bold')
    # axs[1].set_zticklabels([-1, -0.5, 0, 0.5, 1], fontweight='bold')
    
    # Show the plot
    plt.tight_layout()
    #plt.savefig(f"plot{index}.png", dpi=500)
    plt.show()

def unscale_pc(pc_scaled):
    """[x/250, y/250, z/500+1] -> [x,y,z] in mm."""
    pts = pc_scaled.clone()
    pts[:, :, 0] = pts[:, :, 0] * 250.0  # Changed from pts[:, 0]
    pts[:, :, 1] = pts[:, :, 1] * 250.0  # Changed from pts[:, 1]
    pts[:, :, 2] = (pts[:, :, 2] - 1.0) * 500.0  # Changed from pts[:, 2]
    pts[:, :, 3] = torch.pow(10, pts[:, :, 3])  # Unlog base-10
    
    return pts


                #####################################
                ###### Translate: SC ---> no SC #####
                #####################################

all_orig = []
all_trans = []

with torch.no_grad():
    for i, batch in enumerate(sim_loader):
        data = batch[0].to(device)
        mu_sim, sigma_sim = model_sim.encoder(data)
        context_sim = reparameterize_gaussian(mean=mu_sim, logvar=sigma_sim)

        mu_exp, sigma_exp = model_exp.encoder(data)
        context_exp = reparameterize_gaussian(mean=mu_exp, logvar=sigma_exp)

        x_T, eps = dpm_encoder(data, model_sim, context_sim)
        x_T = x_T.to(device)
        eps = eps.to(device)
        print("successfully encoded")
        x = sample(model_exp, 512, context_exp, x_T, eps)

        orig = unscale_pc(data).cpu()
        trans = unscale_pc(x).cpu()
        
        all_orig.append(orig)
        all_trans.append(trans)
        
orig = torch.cat(all_orig, dim=0)
trans = torch.cat(all_trans, dim=0)

np.save('original.npy', orig.numpy())  # shape (N, 512, 4)
np.save('translated.npy', trans.numpy())  # shape (N, 512, 4)