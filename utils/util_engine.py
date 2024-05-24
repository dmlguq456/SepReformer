import os
import torch
import numpy as np
from loguru import logger

from torchinfo import summary as summary_
from ptflops import get_model_complexity_info
from thop import profile

import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import numpy as np
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
import seaborn as sns
import torch
from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler
from scipy.stats import zscore
import pandas as pd
from scipy.spatial.distance import cosine



def load_last_checkpoint_n_get_epoch(checkpoint_dir, model, optimizer, location):
    """
    Load the latest checkpoint (model state and optimizer state) from a given directory.

    Args:
        checkpoint_dir (str): Directory containing the checkpoint files.
        model (torch.nn.Module): The model into which the checkpoint's model state should be loaded.
        optimizer (torch.optim.Optimizer): The optimizer into which the checkpoint's optimizer state should be loaded.
        location (str, optional): Device location for loading the checkpoint. Defaults to 'cpu'.

    Returns:
        int: The epoch number associated with the loaded checkpoint. 
             If no checkpoint is found, returns 0 as the starting epoch.

    Notes:
        - The checkpoint file is expected to have keys: 'model_state_dict', 'optimizer_state_dict', and 'epoch'.
        - If there are multiple checkpoint files in the directory, the one with the highest epoch number is loaded.
    """
    # List all .pkl files in the directory
    checkpoint_files = [f for f in os.listdir(checkpoint_dir)]

    # If there are no checkpoint files, return 0 as the starting epoch
    if not checkpoint_files: return 1
    else:
        # Extract the epoch numbers from the file names and find the latest (max)
        epochs = [int(f.split('.')[1]) for f in checkpoint_files]
        latest_checkpoint_file = os.path.join(checkpoint_dir, checkpoint_files[epochs.index(max(epochs))])

        # Load the checkpoint into the model & optimizer
        logger.info(f"Loaded Pretrained model from {latest_checkpoint_file} .....")
        checkpoint_dict = torch.load(latest_checkpoint_file, map_location=location)
        model.load_state_dict(checkpoint_dict['model_state_dict'], strict=False) # Depend on weight file's key!!
        optimizer.load_state_dict(checkpoint_dict['optimizer_state_dict'])
        
        # Retrun latent epoch
        return checkpoint_dict['epoch'] + 1
    
def save_checkpoint_per_nth(nth, epoch, model, optimizer, train_loss, valid_loss, checkpoint_path, wandb_run):
    """
    Save the state of the model and optimizer every nth epoch to a checkpoint file.
    Additionally, log and save the checkpoint file using wandb.

    Args:
        nth (int): Interval for which checkpoints should be saved.
        epoch (int): The current training epoch.
        model (nn.Module): The model whose state needs to be saved.
        optimizer (Optimizer): The optimizer whose state needs to be saved.
        checkpoint_path (str): Directory path where the checkpoint will be saved.
        wandb_run (wandb.wandb_run.Run): The current wandb run to log and save the checkpoint.

    Returns:
        None
    """
    if epoch % nth == 0:
        # Save the state of the model and optimizer to a checkpoint file
        torch.save(
                    {
                        'epoch': epoch,
                        'model_state_dict': model.state_dict(),
                        'optimizer_state_dict': optimizer.state_dict(),
                        'train_loss': train_loss,
                        'valid_loss': valid_loss
                    },
                    os.path.join(checkpoint_path, f"epoch.{epoch:04}.pth"))
        
        # Log and save the checkpoint file using wandb
        wandb_run.save(os.path.join(checkpoint_path, f"epoch.{epoch:04}.pth"))

def save_checkpoint_per_best(best, valid_loss, train_loss, epoch, model, optimizer, checkpoint_path, wandb_run):
    """
    Save the state of the model and optimizer every nth epoch to a checkpoint file.
    Additionally, log and save the checkpoint file using wandb.

    Args:
        nth (int): Interval for which checkpoints should be saved.
        epoch (int): The current training epoch.
        model (nn.Module): The model whose state needs to be saved.
        optimizer (Optimizer): The optimizer whose state needs to be saved.
        checkpoint_path (str): Directory path where the checkpoint will be saved.
        wandb_run (wandb.wandb_run.Run): The current wandb run to log and save the checkpoint.

    Returns:
        None
    """
    if valid_loss < best:
        # Save the state of the model and optimizer to a checkpoint file
        torch.save(
                    {
                        'epoch': epoch,
                        'model_state_dict': model.state_dict(),
                        'optimizer_state_dict': optimizer.state_dict(),
                        'train_loss': train_loss,
                        'valid_loss': valid_loss
                    },
                    os.path.join(checkpoint_path, f"epoch.{epoch:04}.pth"))
        
        # # Log and save the checkpoint file using wandb
        # wandb_run.save(os.path.join(checkpoint_path, f"epoch.{epoch:04}.pth"))
        best = valid_loss
    return best

def step_scheduler(scheduler, **kwargs):
    if isinstance(scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
        scheduler.step(kwargs.get('val_loss'))
    elif isinstance(scheduler, torch.optim.lr_scheduler.StepLR):
        scheduler.step()
    elif isinstance(scheduler, torch.optim.lr_scheduler.CosineAnnealingLR):
        scheduler.step()
    # Add another schedulers
    else:
        raise ValueError(f"Unknown scheduler type: {type(scheduler)}")

def print_parameters_count(model):
    total_parameters = 0
    for name, param in model.named_parameters():
        param_count = param.numel()
        total_parameters += param_count
        logger.info(f"{name}: {param_count}")
    logger.info(f"Total parameters: {(total_parameters / 1e6):.2f}M")

def model_params_mac_summary(model, input, dummy_input, metrics):
    
    # ptflpos
    if 'ptflops' in metrics:
        MACs_ptflops, params_ptflops = get_model_complexity_info(model, (input.shape[1],)) # (num_samples,)
        MACs_ptflops, params_ptflops = MACs_ptflops.replace(" MMac", ""), params_ptflops.replace(" M", "")
        logger.info(f"ptflops: MMac: {MACs_ptflops}, Params: {params_ptflops}")

    # thop
    if 'thop' in metrics:
        MACs_thop, params_thop = profile(model, inputs=(input, ), verbose=False)
        MACs_thop, params_thop = MACs_thop/1e6, params_thop/1e6
        logger.info(f"thop: MMac: {MACs_thop}, Params: {params_thop}")
    
    # torchinfo
    if 'torchinfo' in metrics:
        model_profile = summary_(model, input_size=input.size())
        MACs_torchinfo, params_torchinfo = model_profile.total_mult_adds/1e6, model_profile.total_params/1e6
        logger.info(f"torchinfo: MMac: {MACs_torchinfo}, Params: {params_torchinfo}")


    # MEASURE PERFORMANCE
    starter, ender = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
    repetitions = 500
    repetitions2 = 500
    timings = np.zeros((repetitions,1))
    torch.set_num_threads(1)
    with torch.no_grad():
        for rep in range(repetitions+repetitions2):
            if rep > repetitions:
                starter.record()
                _ = model(dummy_input)
                ender.record()
                # WAIT FOR GPU SYNC
                torch.cuda.synchronize()
                curr_time = starter.elapsed_time(ender)
                timings[rep-repetitions2] = curr_time
    logger.info(f"Timing: {timings.mean()}")




def normalization_with_outliers(data, threshold=3.0):
    # Z-score 계산
    z_scores = zscore(data, axis=0)
    
    # 아웃라이어 마스크 생성
    mask = np.abs(z_scores) < threshold
    
    # 각 피처별로 아웃라이어를 제외한 데이터 설정
    filtered_data = np.copy(data)
    filtered_data[~mask] = np.nan

    # 가우시안 정규화 (표준화)
    scaler = StandardScaler()
    normalized_data = np.zeros_like(data)

    # 각 피처별로 정규화
    for i in range(data.shape[1]):
        feature_data = filtered_data[:, i]
        valid_data = feature_data[~np.isnan(feature_data)]
        
        if len(valid_data) > 0:
            valid_data = valid_data.reshape(-1, 1)
            scaler.fit(valid_data)
            normalized_column = scaler.transform(feature_data.reshape(-1, 1)).flatten()
            
            # 아웃라이어 값을 -1 또는 1로 설정
            normalized_column[np.isnan(normalized_column)] = np.where(z_scores[:, i][~mask[:, i]] > 0, 1, -1)
            normalized_data[:, i] = normalized_column
        else:
            normalized_data[:, i] = np.nan

    return normalized_data


def create_combined_heatmap(data1, data2, scale_factor=8):
    diff = data1 - data2
    combined = np.zeros((data1.shape[0], data1.shape[1], 3))

    max_abs_diff = np.max(np.abs(diff))
    norm_diff = diff / max_abs_diff
    scaled_diff = norm_diff * scale_factor

    combined[:, :, 0] = np.clip(scaled_diff, 0, 1)  # Red channel
    combined[:, :, 2] = np.clip(-scaled_diff, 0, 1)  # Blue channel

    return combined
    
def visualization_in_one_batch(x):
    if hasattr(visualization_in_one_batch, 'call_count'):
        visualization_in_one_batch.call_count += 1
    else:
        visualization_in_one_batch.call_count = 1
    
    x = x.cpu().detach()  # x'shape=[2, F, T]
    
    # 각 [F, T] 데이터에 대해 정규화
    data_speaker_1 = x[0].numpy().T  # [T, F]
    data_speaker_2 = x[1].numpy().T  # [T, F]

    # PCA 및 t-SNE 객체 생성
    pca_3d = PCA(n_components=3)
    pca_2d = PCA(n_components=2)
    pca_1d = PCA(n_components=1)
    tsne_3d = TSNE(n_components=3, perplexity=30, n_iter=300)
    tsne_2d = TSNE(n_components=2, perplexity=30, n_iter=300)
    tsne_1d = TSNE(n_components=1, perplexity=30, n_iter=300)

    # 3D, 2D 및 1D PCA 변환
    pca_result_3d_1 = pca_3d.fit_transform(data_speaker_1)
    pca_result_3d_2 = pca_3d.fit_transform(data_speaker_2)
    pca_result_2d_1 = pca_2d.fit_transform(data_speaker_1)
    pca_result_2d_2 = pca_2d.fit_transform(data_speaker_2)
    pca_result_1d_1 = pca_1d.fit_transform(data_speaker_1).flatten()
    pca_result_1d_2 = pca_1d.fit_transform(data_speaker_2).flatten()

    # 3D, 2D 및 1D t-SNE 변환
    tsne_result_3d_1 = tsne_3d.fit_transform(data_speaker_1)
    tsne_result_3d_2 = tsne_3d.fit_transform(data_speaker_2)
    tsne_result_2d_1 = tsne_2d.fit_transform(data_speaker_1)
    tsne_result_2d_2 = tsne_2d.fit_transform(data_speaker_2)
    tsne_result_1d_1 = tsne_1d.fit_transform(data_speaker_1).flatten()
    tsne_result_1d_2 = tsne_1d.fit_transform(data_speaker_2).flatten()

    # 데이터 정규화 후 오름차순 정렬 (히트맵용)
    data_speaker_concat = np.concatenate((data_speaker_1, data_speaker_2))
    sorted_data_speaker_concat = np.sort(data_speaker_concat, axis=1)
    sorted_data_speaker_1, sorted_data_speaker_2 = np.split(sorted_data_speaker_concat, 2, axis=0)
    normd_sorted_data_speaker_1 = normalization_with_outliers(sorted_data_speaker_1)
    normd_sorted_data_speaker_2 = normalization_with_outliers(sorted_data_speaker_2)
    
    # Cosine similarity 계산
    cosine_similarity = np.array([1 - cosine(data_speaker_1[i], data_speaker_2[i]) for i in range(len(data_speaker_1))])
    
    # Combine heatmap
    combined_heatmap = create_combined_heatmap(sorted_data_speaker_1.T, sorted_data_speaker_2.T)
    
    # l2_norm_diff, combined_heatmap, Heatmap of Data 1, Heatmap of Data 2 저장
    cosine_similarity_df = pd.DataFrame(cosine_similarity, columns=['Cosine Similarity'])
    
    cosine_similarity_df.to_csv(f"stage_{visualization_in_one_batch.call_count}_cosine_similarity.csv", index=False)


    # 시각화
    fig = plt.figure(figsize=(72, 24))
    gs = fig.add_gridspec(2, 6)

    # 3D PCA 시각화
    ax = fig.add_subplot(gs[0, 0], projection='3d')
    ax.scatter(pca_result_3d_1[:, 0], pca_result_3d_1[:, 1], pca_result_3d_1[:, 2], c='red', label='Data 1')
    ax.scatter(pca_result_3d_2[:, 0], pca_result_3d_2[:, 1], pca_result_3d_2[:, 2], c='blue', label='Data 2')
    ax.set_title('3D PCA Results')
    ax.set_xlabel('PC1')
    ax.set_ylabel('PC2')
    ax.set_zlabel('PC3')
    ax.legend()

    # 2D PCA 시각화
    ax = fig.add_subplot(gs[0, 1])
    ax.scatter(pca_result_2d_1[:, 0], pca_result_2d_1[:, 1], c='red', label='Data 1')
    ax.scatter(pca_result_2d_2[:, 0], pca_result_2d_2[:, 1], c='blue', label='Data 2')
    ax.set_title('2D PCA Results')
    ax.set_xlabel('PC1')
    ax.set_ylabel('PC2')
    ax.legend()

    # 1D PCA 시각화
    ax = fig.add_subplot(gs[0, 2])
    ax.scatter(pca_result_1d_1, np.zeros_like(pca_result_1d_1), c='red', alpha=0.5, label='Data 1')
    ax.scatter(pca_result_1d_2, np.zeros_like(pca_result_1d_2), c='blue', alpha=0.5, label='Data 2')
    ax.set_title('1D PCA Results')
    ax.set_xlabel('PC1')
    ax.set_yticks([])  # Hide y-axis
    ax.legend()

    # 3D t-SNE 시각화
    ax = fig.add_subplot(gs[0, 3], projection='3d')
    ax.scatter(tsne_result_3d_1[:, 0], tsne_result_3d_1[:, 1], tsne_result_3d_1[:, 2], c='red', label='Data 1')
    ax.scatter(tsne_result_3d_2[:, 0], tsne_result_3d_2[:, 1], tsne_result_3d_2[:, 2], c='blue', label='Data 2')
    ax.set_title('3D t-SNE Results')
    ax.set_xlabel('t-SNE1')
    ax.set_ylabel('t-SNE2')
    ax.set_zlabel('t-SNE3')
    ax.legend()

    # 2D t-SNE 시각화
    ax = fig.add_subplot(gs[0, 4])
    ax.scatter(tsne_result_2d_1[:, 0], tsne_result_2d_1[:, 1], c='red', label='Data 1')
    ax.scatter(tsne_result_2d_2[:, 0], tsne_result_2d_2[:, 1], c='blue', label='Data 2')
    ax.set_title('2D t-SNE Results')
    ax.set_xlabel('t-SNE1')
    ax.set_ylabel('t-SNE2')
    ax.legend()

    # 1D t-SNE 시각화
    ax = fig.add_subplot(gs[0, 5])
    ax.scatter(tsne_result_1d_1, np.zeros_like(tsne_result_1d_1), c='red', alpha=0.5, label='Data 1')
    ax.scatter(tsne_result_1d_2, np.zeros_like(tsne_result_1d_2), c='blue', alpha=0.5, label='Data 2')
    ax.set_title('1D t-SNE Results')
    ax.set_xlabel('t-SNE1')
    ax.set_yticks([])  # Hide y-axis
    ax.legend()

    # Heatmap of features (by speaker)
    ax = fig.add_subplot(gs[1, :2])
    sns.heatmap(normd_sorted_data_speaker_1.T, cmap="jet", ax=ax, vmin=-2.5, vmax=2.5)
    ax.set_title('Heatmap of Data 1')

    ax = fig.add_subplot(gs[1, 2:4])
    sns.heatmap(normd_sorted_data_speaker_2.T, cmap="jet", ax=ax, vmin=-2.5, vmax=2.5)
    ax.set_title('Heatmap of Data 2')
    
    # Combine Heatmap
    ax = fig.add_subplot(gs[1, 4:])
    ax.imshow(combined_heatmap, aspect='auto')
    ax.set_title('Combined Heatmap')
    


    save_path = f"split_{visualization_in_one_batch.call_count}.png"
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close(fig)

def visualization_in_one_batch_inner_stage(x):
    if hasattr(visualization_in_one_batch_inner_stage, 'call_count'):
        visualization_in_one_batch_inner_stage.call_count += 1
    else:
        visualization_in_one_batch_inner_stage.call_count = 1
    
    x = x.cpu().detach()  # x'shape=[2, F, T]
    
    # 각 [F, T] 데이터에 대해 정규화
    data_speaker_1 = x[0].numpy().T  # [T, F]
    data_speaker_2 = x[1].numpy().T  # [T, F]

    # PCA 및 t-SNE 객체 생성
    pca_3d = PCA(n_components=3)
    pca_2d = PCA(n_components=2)
    pca_1d = PCA(n_components=1)
    tsne_3d = TSNE(n_components=3, perplexity=30, n_iter=300)
    tsne_2d = TSNE(n_components=2, perplexity=30, n_iter=300)
    tsne_1d = TSNE(n_components=1, perplexity=30, n_iter=300)

    # 3D, 2D 및 1D PCA 변환
    pca_result_3d_1 = pca_3d.fit_transform(data_speaker_1)
    pca_result_3d_2 = pca_3d.fit_transform(data_speaker_2)
    pca_result_2d_1 = pca_2d.fit_transform(data_speaker_1)
    pca_result_2d_2 = pca_2d.fit_transform(data_speaker_2)
    pca_result_1d_1 = pca_1d.fit_transform(data_speaker_1).flatten()
    pca_result_1d_2 = pca_1d.fit_transform(data_speaker_2).flatten()

    # 3D, 2D 및 1D t-SNE 변환
    tsne_result_3d_1 = tsne_3d.fit_transform(data_speaker_1)
    tsne_result_3d_2 = tsne_3d.fit_transform(data_speaker_2)
    tsne_result_2d_1 = tsne_2d.fit_transform(data_speaker_1)
    tsne_result_2d_2 = tsne_2d.fit_transform(data_speaker_2)
    tsne_result_1d_1 = tsne_1d.fit_transform(data_speaker_1).flatten()
    tsne_result_1d_2 = tsne_1d.fit_transform(data_speaker_2).flatten()

    # 데이터 정규화 후 오름차순 정렬 (히트맵용)
    data_speaker_concat = np.concatenate((data_speaker_1, data_speaker_2))
    sorted_data_speaker_concat = np.sort(data_speaker_concat, axis=1)
    sorted_data_speaker_1, sorted_data_speaker_2 = np.split(sorted_data_speaker_concat, 2, axis=0)
    normd_sorted_data_speaker_1 = normalization_with_outliers(sorted_data_speaker_1)
    normd_sorted_data_speaker_2 = normalization_with_outliers(sorted_data_speaker_2)

    # Cosine similarity 계산
    cosine_similarity = np.array([1 - cosine(data_speaker_1[i], data_speaker_2[i]) for i in range(len(data_speaker_1))])
    
    # Combine heatmap
    combined_heatmap = create_combined_heatmap(normd_sorted_data_speaker_1.T, normd_sorted_data_speaker_2.T)
    

    cosine_similarity_df = pd.DataFrame(cosine_similarity, columns=['Cosine Similarity'])
    cosine_similarity_df.to_csv(f"stage_{visualization_in_one_batch.call_count}_block_out_{visualization_in_one_batch_inner_stage.call_count}_cosine_similarity.csv", index=False)

    # 시각화
    fig = plt.figure(figsize=(36, 12))
    gs = fig.add_gridspec(2, 6)

    # 3D PCA 시각화
    ax = fig.add_subplot(gs[0, 0], projection='3d')
    ax.scatter(pca_result_3d_1[:, 0], pca_result_3d_1[:, 1], pca_result_3d_1[:, 2], c='red', label='Data 1')
    ax.scatter(pca_result_3d_2[:, 0], pca_result_3d_2[:, 1], pca_result_3d_2[:, 2], c='blue', label='Data 2')
    ax.set_title('3D PCA Results')
    ax.set_xlabel('PC1')
    ax.set_ylabel('PC2')
    ax.set_zlabel('PC3')
    ax.legend()

    # 2D PCA 시각화
    ax = fig.add_subplot(gs[0, 1])
    ax.scatter(pca_result_2d_1[:, 0], pca_result_2d_1[:, 1], c='red', label='Data 1')
    ax.scatter(pca_result_2d_2[:, 0], pca_result_2d_2[:, 1], c='blue', label='Data 2')
    ax.set_title('2D PCA Results')
    ax.set_xlabel('PC1')
    ax.set_ylabel('PC2')
    ax.legend()

    # 1D PCA 시각화
    ax = fig.add_subplot(gs[0, 2])
    ax.scatter(pca_result_1d_1, np.zeros_like(pca_result_1d_1), c='red', alpha=0.5, label='Data 1')
    ax.scatter(pca_result_1d_2, np.zeros_like(pca_result_1d_2), c='blue', alpha=0.5, label='Data 2')
    ax.set_title('1D PCA Results')
    ax.set_xlabel('PC1')
    ax.set_yticks([])  # Hide y-axis
    ax.legend()

    # 3D t-SNE 시각화
    ax = fig.add_subplot(gs[0, 3], projection='3d')
    ax.scatter(tsne_result_3d_1[:, 0], tsne_result_3d_1[:, 1], tsne_result_3d_1[:, 2], c='red', label='Data 1')
    ax.scatter(tsne_result_3d_2[:, 0], tsne_result_3d_2[:, 1], tsne_result_3d_2[:, 2], c='blue', label='Data 2')
    ax.set_title('3D t-SNE Results')
    ax.set_xlabel('t-SNE1')
    ax.set_ylabel('t-SNE2')
    ax.set_zlabel('t-SNE3')
    ax.legend()

    # 2D t-SNE 시각화
    ax = fig.add_subplot(gs[0, 4])
    ax.scatter(tsne_result_2d_1[:, 0], tsne_result_2d_1[:, 1], c='red', label='Data 1')
    ax.scatter(tsne_result_2d_2[:, 0], tsne_result_2d_2[:, 1], c='blue', label='Data 2')
    ax.set_title('2D t-SNE Results')
    ax.set_xlabel('t-SNE1')
    ax.set_ylabel('t-SNE2')
    ax.legend()

    # 1D t-SNE 시각화
    ax = fig.add_subplot(gs[0, 5])
    ax.scatter(tsne_result_1d_1, np.zeros_like(tsne_result_1d_1), c='red', alpha=0.5, label='Data 1')
    ax.scatter(tsne_result_1d_2, np.zeros_like(tsne_result_1d_2), c='blue', alpha=0.5, label='Data 2')
    ax.set_title('1D t-SNE Results')
    ax.set_xlabel('t-SNE1')
    ax.set_yticks([])  # Hide y-axis
    ax.legend()

    # Heatmap of features (by speaker)
    ax = fig.add_subplot(gs[1, :2])
    sns.heatmap(normd_sorted_data_speaker_1.T, cmap="jet", ax=ax, vmin=-2.5, vmax=2.5)
    ax.set_title('Heatmap of Data 1')

    ax = fig.add_subplot(gs[1, 2:4])
    sns.heatmap(normd_sorted_data_speaker_2.T, cmap="jet", ax=ax, vmin=-2.5, vmax=2.5)
    ax.set_title('Heatmap of Data 2')
    
    # Combine Heatmap
    ax = fig.add_subplot(gs[1, 4:])
    ax.imshow(combined_heatmap, aspect='auto')
    ax.set_title('Combined Heatmap')


    save_path = f"stage_{visualization_in_one_batch.call_count}_block_out_{visualization_in_one_batch_inner_stage.call_count}.png"
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close(fig)


def visualization_in_one_batch_for_late(x):
    if hasattr(visualization_in_one_batch_for_late, 'call_count'):
        visualization_in_one_batch_for_late.call_count += 1
    else:
        visualization_in_one_batch_for_late.call_count = 1
    
    x = x.cpu().detach()  # x'shape=[1, F, T]
    
    # [F, T] 데이터에 대해 정규화
    data = x[0].numpy().T  # [T, F]

    # 데이터 정규화 및 오름차순 정렬 (히트맵용)
    normalized_data = normalization_with_outliers(data)
    sorted_data = np.sort(normalized_data, axis=1)  # 각 시간별로 채널 값 오름차순 정렬
    

    # PCA 및 t-SNE 객체 생성
    pca_3d = PCA(n_components=3)
    pca_2d = PCA(n_components=2)
    pca_1d = PCA(n_components=1)
    tsne_3d = TSNE(n_components=3, perplexity=30, n_iter=300)
    tsne_2d = TSNE(n_components=2, perplexity=30, n_iter=300)
    tsne_1d = TSNE(n_components=1, perplexity=30, n_iter=300)

    # 3D, 2D 및 1D PCA 변환
    pca_result_3d = pca_3d.fit_transform(data)
    pca_result_2d = pca_2d.fit_transform(data)
    pca_result_1d = pca_1d.fit_transform(data).flatten()

    # 3D, 2D 및 1D t-SNE 변환
    tsne_result_3d = tsne_3d.fit_transform(data)
    tsne_result_2d = tsne_2d.fit_transform(data)
    tsne_result_1d = tsne_1d.fit_transform(data).flatten()

    # Flatten data for histogram (original data)
    flattened_data = data.flatten()

    # 시각화
    fig = plt.figure(figsize=(72, 24))
    gs = fig.add_gridspec(2, 6)

    # 3D PCA 시각화
    ax = fig.add_subplot(gs[0, 0], projection='3d')
    ax.scatter(pca_result_3d[:, 0], pca_result_3d[:, 1], pca_result_3d[:, 2], c='green')
    ax.set_title('3D PCA Results')
    ax.set_xlabel('PC1')
    ax.set_ylabel('PC2')
    ax.set_zlabel('PC3')

    # 2D PCA 시각화
    ax = fig.add_subplot(gs[0, 1])
    ax.scatter(pca_result_2d[:, 0], pca_result_2d[:, 1], c='green')
    ax.set_title('2D PCA Results')
    ax.set_xlabel('PC1')
    ax.set_ylabel('PC2')

    # 1D PCA 시각화
    ax = fig.add_subplot(gs[0, 2])
    ax.scatter(pca_result_1d, np.zeros_like(pca_result_1d), c='green', alpha=0.5)
    ax.set_title('1D PCA Results')
    ax.set_xlabel('PC1')
    ax.set_yticks([])  # Hide y-axis

    # 3D t-SNE 시각화
    ax = fig.add_subplot(gs[0, 3], projection='3d')
    ax.scatter(tsne_result_3d[:, 0], tsne_result_3d[:, 1], tsne_result_3d[:, 2], c='green')
    ax.set_title('3D t-SNE Results')
    ax.set_xlabel('t-SNE1')
    ax.set_ylabel('t-SNE2')
    ax.set_zlabel('t-SNE3')

    # 2D t-SNE 시각화
    ax = fig.add_subplot(gs[0, 4])
    ax.scatter(tsne_result_2d[:, 0], tsne_result_2d[:, 1], c='green')
    ax.set_title('2D t-SNE Results')
    ax.set_xlabel('t-SNE1')
    ax.set_ylabel('t-SNE2')

    # 1D t-SNE 시각화
    ax = fig.add_subplot(gs[0, 5])
    ax.scatter(tsne_result_1d, np.zeros_like(tsne_result_1d), c='green', alpha=0.5)
    ax.set_title('1D t-SNE Results')
    ax.set_xlabel('t-SNE1')
    ax.set_yticks([])  # Hide y-axis

    # Heatmap of features
    ax = fig.add_subplot(gs[1, :3])
    sns.heatmap(sorted_data.T, cmap="viridis", ax=ax)
    ax.set_title('Heatmap of Data')

    # Flattened histogram
    ax = fig.add_subplot(gs[1, 3:])
    ax.hist(flattened_data, bins=100, color='green', alpha=0.5)
    ax.set_title('Flattened Histogram of All Features')

    save_path = f"late_{visualization_in_one_batch_for_late.call_count}.png"
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close(fig)
    
def visualization_attention_map(attention_attr):
    if hasattr(visualization_attention_map, 'call_count'):
        visualization_attention_map.call_count += 1
    else:
        visualization_attention_map.call_count = 1
        
    # Get the attention weights
    attention_map = attention_attr.cpu().detach().numpy()  # (batch, head, attention_x, attention_y)
    
    # shape: (batch, head, attention_x, attention_y)
    batch_size, num_heads, attention_x, attention_y = attention_map.shape
    
    # head, attention_x, attention_y 차원을 결합하여 y값들을 동일한 차원으로 만듭니다.
    attention_combined = attention_map.reshape(batch_size, num_heads * attention_x * attention_y)  # shape: (batch, head * attention_x * attention_y)

    # 값을 반복하여 격자 모양을 만들기 위해 각 attention index를 반복합니다.
    repeat_factor_for_resolution = attention_x * attention_y * 128
    attention_combined_repeated = np.repeat(attention_combined, repeat_factor_for_resolution, axis=1)
    
    attention_combined_repeated = attention_combined_repeated.T  # (head * attention_x * attention_y * repeat_factor, batch)
    
    # 플롯 초기화
    plt.figure()

    # 2D 히트맵 그리기 (전체 시퀀스를 x축으로)
    img = plt.imshow(attention_combined_repeated, aspect='auto', cmap='gray', vmin=0, vmax=1)
    cbar = plt.colorbar(img)
    cbar.set_label('Attention score')
    plt.xlabel('Frame time index')
    plt.ylabel('Attention index')
    plt.title('2D Attention Heatmap')
    
    # y축 눈금을 repeat_factor로 나눈 값으로 설정
    y_ticks = np.arange(0, attention_combined_repeated.shape[0], repeat_factor_for_resolution)
    y_tick_labels = (y_ticks // repeat_factor_for_resolution)
    plt.yticks(y_ticks, y_tick_labels)
    
    # y값이 (attention_x * attention_y)의 배수일 때 x축 방향으로 빨간색 선을 그리기
    for y in range(0, attention_combined_repeated.shape[0], repeat_factor_for_resolution*attention_x*attention_y):
        if y != 0: plt.axhline(y=y, color='red', linestyle='-', linewidth=2)
    
    # 어텐션맵을 파일로 저장
    save_path = f"stage_{visualization_in_one_batch.call_count}_cs_attn_{visualization_attention_map.call_count}.png"
    plt.savefig(save_path)