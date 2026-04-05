import os, warnings, gc
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

# ─── 1. CLEAN TPU SETUP ───────────────────────────────────────────
os.environ.pop('TPU_PROCESS_ADDRESSES', None)
os.environ.pop('CLOUD_TPU_TASK_ID', None)
os.environ['XLA_USE_BF16'] = '1'
import torch_xla.core.xla_model as xm
import torch_xla.distributed.parallel_loader as pl
import torch_xla.distributed.xla_multiprocessing as xmp
import torch_xla.runtime as xr 
warnings.filterwarnings('ignore')

# ─── 2. CONFIGURATION ─────────────────────────────────────────────
def find_comp_path():
    for root, dirs, files in os.walk('/kaggle/input'):
        if 'lat_long.npy' in files: return os.path.dirname(root)
    return None

BASE_PATH = find_comp_path()
RAW_BASE  = os.path.join(BASE_PATH, 'raw')
TEST_BASE = os.path.join(BASE_PATH, 'test_in')

SEED       = 123
EPOCHS     = 50   
STRIDE     = 1          
BATCH_SIZE = 64        # Reduced to 64 to save VRAM for the "Genius" Attention math
WD         = 1e-2       
MONTHS     = ['APRIL_16', 'JULY_16', 'OCT_16', 'DEC_16'] # The 4 available months
ALL_VARS   = ['q2','t2','u10','v10','swdown','pblh','psfc','rain','PM25','NH3','SO2','NOx','NMVOC_e','NMVOC_finn','bio']

# ─── 3. GLOBAL NORMALIZATION (The "Generalist" Fix) ───────────────
def get_stats():
    # Hardcoded to prevent overfitting to just April
    return {
        'cpm25': (65.0, 50.0), 'q2': (0.015, 0.005), 't2': (300.0, 15.0),
        'u10': (0.0, 5.0), 'v10': (0.0, 5.0), 'swdown': (400.0, 300.0),
        'pblh': (800.0, 600.0), 'psfc': (98000.0, 2500.0), 'rain': (0.001, 0.05)
    }
norm_stats = get_stats()
def normalize(arr, v):
    m, s = norm_stats.get(v, (0.0, 1.0))
    return (arr - m) / (s + 1e-8)
def denormalize(arr): return arr * norm_stats['cpm25'][1] + norm_stats['cpm25'][0]

# ─── 4. THE PRODIGY ARCHITECTURE ──────────────────────────────────
class AdvectionDiffusionCell(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.diffusion = nn.Conv2d(channels, channels, kernel_size=3, padding=1, groups=channels)
        self.mix = nn.Conv2d(channels * 2, channels, kernel_size=1)

    def forward(self, x, wind_u, wind_v):
        B, C, H, W = x.shape
        diffused = self.diffusion(x)
        
        grid_y, grid_x = torch.meshgrid(torch.linspace(-1, 1, H), torch.linspace(-1, 1, W), indexing='ij')
        grid = torch.stack((grid_x, grid_y), dim=-1).unsqueeze(0).expand(B, -1, -1, -1).to(x.device)
        
        wind_flow = torch.stack((wind_u, wind_v), dim=-1) * 0.05 
        advection_grid = grid - wind_flow 
        
        advected = F.grid_sample(x, advection_grid, align_corners=True, padding_mode='border')
        return F.gelu(self.mix(torch.cat([diffused, advected], dim=1)))

class GlobalNodeAttention(nn.Module):
    def __init__(self, dim, heads=8):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.norm2 = nn.LayerNorm(dim)
        self.attn = nn.MultiheadAttention(embed_dim=dim, num_heads=heads, batch_first=True)
        self.mlp = nn.Sequential(
            nn.Linear(dim, dim * 2), nn.GELU(), nn.Dropout(0.1), nn.Linear(dim * 2, dim)
        )

    def forward(self, x):
        norm_x = self.norm1(x)
        attn_out, _ = self.attn(norm_x, norm_x, norm_x)
        x = x + attn_out
        x = x + self.mlp(self.norm2(x))
        return x

class GeniusChildNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.feat = nn.Sequential(
            nn.Conv2d(27, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64), nn.GELU(),
            nn.Conv2d(64, 128, kernel_size=3, padding=1)
        )
        self.physics = AdvectionDiffusionCell(128)
        self.patchify = nn.Conv2d(128, 256, kernel_size=10, stride=10) 
        self.global_reaction = nn.Sequential(*[GlobalNodeAttention(256) for _ in range(3)])
        self.unpatchify = nn.ConvTranspose2d(256, 128, kernel_size=10, stride=10)
        self.out = nn.Sequential(
            nn.Conv2d(256, 64, kernel_size=3, padding=1), 
            nn.GELU(),
            nn.Conv2d(64, 16, kernel_size=1)
        )

    def forward(self, pm, aux, pos):
        x = torch.cat([pm, aux, pos], dim=1)
        x = F.pad(x, (0, 16, 0, 0)) 
        
        wind_u, wind_v = aux[:, 2, :, :], aux[:, 3, :, :]
        wind_u = F.pad(wind_u, (0, 16, 0, 0))
        wind_v = F.pad(wind_v, (0, 16, 0, 0))
        
        h_base = self.feat(x)
        h_phys = self.physics(h_base, wind_u, wind_v)
        
        nodes = self.patchify(h_phys).flatten(2).transpose(1, 2)
        nodes = self.global_reaction(nodes)
        
        h_global = self.unpatchify(nodes.transpose(1, 2).reshape(-1, 256, 14, 14))
        
        out = self.out(torch.cat([h_phys, h_global], dim=1))
        return out[:, :, :, :124]

# ─── 5. TPU WORKER & RELAY-RACE CHECKPOINTING ─────────────────────
def _mp_fn(index):
    torch.manual_seed(SEED)
    device = xm.xla_device()
    
    latlon = np.load(f'{RAW_BASE}/lat_long.npy')
    pos_enc = torch.from_numpy(np.stack([
        (latlon[:,:,0]-latlon[:,:,0].min())/(latlon[:,:,0].max()-latlon[:,:,0].min()+1e-8),
        (latlon[:,:,1]-latlon[:,:,1].min())/(latlon[:,:,1].max()-latlon[:,:,1].min()+1e-8)
    ], axis=0).astype(np.float32)).unsqueeze(0)

    model = GeniusChildNet().to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=WD)

    class Phase2Dataset(Dataset):
        def __init__(self):
            self.samples = []
            for m in MONTHS:
                t_limit = np.load(f'{RAW_BASE}/{m}/cpm25.npy', mmap_mode='r').shape[0] - 26
                for t in range(0, t_limit, STRIDE): self.samples.append((m, t))
        def __len__(self): return len(self.samples)
        def __getitem__(self, idx):
            m, t = self.samples[idx]
            pm_in = normalize(np.load(f'{RAW_BASE}/{m}/cpm25.npy', mmap_mode='r')[t:t+10], 'cpm25')
            aux = np.stack([normalize(np.load(f'{RAW_BASE}/{m}/{v}.npy', mmap_mode='r')[t+9], v) for v in ALL_VARS])
            tgt = normalize(np.load(f'{RAW_BASE}/{m}/cpm25.npy', mmap_mode='r')[t+10:t+26], 'cpm25')
            return torch.from_numpy(pm_in), torch.from_numpy(aux), torch.from_numpy(tgt)

    train_sampler = torch.utils.data.distributed.DistributedSampler(
        Phase2Dataset(), num_replicas=xr.world_size(), rank=xr.global_ordinal(), shuffle=True)
    train_loader = DataLoader(Phase2Dataset(), batch_size=BATCH_SIZE // xr.world_size(), sampler=train_sampler)

    start_epoch = 1
    best_loss = float('inf')
    checkpoint_path = '/kaggle/working/genius_checkpoint.pt'
    
    # RELAY RACE RESUME LOGIC
    if os.path.exists(checkpoint_path):
        if xm.is_master_ordinal(): print("🔄 Resuming training from previous checkpoint...")
        checkpoint = torch.load(checkpoint_path)
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        start_epoch = checkpoint['epoch'] + 1
        best_loss = checkpoint['best_loss']

    for epoch in range(start_epoch, EPOCHS + 1):
        lr = 1e-5 + 0.5 * (1e-3 - 1e-5) * (1 + np.cos(np.pi * epoch / EPOCHS))
        for pg in optimizer.param_groups: pg['lr'] = lr
        
        model.train()
        para_loader = pl.ParallelLoader(train_loader, [device])
        epoch_loss, steps = 0.0, 0
        
        for pm, aux, tgt in para_loader.per_device_loader(device):
            optimizer.zero_grad()
            pos = pos_enc.expand(pm.shape[0], -1, -1, -1).to(device)
            loss = F.huber_loss(model(pm, aux, pos), tgt) 
            loss.backward()
            xm.optimizer_step(optimizer)
            
            epoch_loss += loss.item()
            steps += 1
            
        avg_loss = epoch_loss / steps
        
        if xm.is_master_ordinal():
            star = "⭐" if avg_loss < best_loss else ""
            if avg_loss < best_loss:
                best_loss = avg_loss
                xm.save(model.state_dict(), 'best_physics_model.pt')
            
            # Save the full state to survive Kaggle disconnects
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'best_loss': best_loss
            }, checkpoint_path)
            
            print(f'Ep {epoch:2d}/{EPOCHS} | Loss: {avg_loss:.4f}{star}')

# ─── 6. INFERENCE ─────────────────────────────────────────────────
if __name__ == '__main__':
    print("🧠 Booting The Genius Child (Unified Architecture) on TPU...")
    xmp.spawn(_mp_fn, args=(), nprocs=None, start_method='fork')
    
    print("\n🔮 Starting Inference Phase...")
    device_inf = torch.device('cpu')
    model = GeniusChildNet().to(device_inf)
    model.load_state_dict(torch.load('best_physics_model.pt', map_location=device_inf))
    model.eval()

    latlon = np.load(f'{RAW_BASE}/lat_long.npy')
    pos_enc = torch.from_numpy(np.stack([
        (latlon[:,:,0]-latlon[:,:,0].min())/(latlon[:,:,0].max()-latlon[:,:,0].min()+1e-8),
        (latlon[:,:,1]-latlon[:,:,1].min())/(latlon[:,:,1].max()-latlon[:,:,1].min()+1e-8)
    ], axis=0).astype(np.float32)).unsqueeze(0).to(device_inf)

    test_pm_raw = np.load(os.path.join(TEST_BASE, 'cpm25.npy'), mmap_mode='r')
    all_preds = []

    for i in range(218):
        pm_i = test_pm_raw[i]
        if pm_i.ndim == 4: pm_i = pm_i.squeeze(0)
        pm_t = torch.from_numpy(normalize(pm_i.astype(np.float32), 'cpm25')).unsqueeze(0).to(device_inf) 

        aux_list = []
        for v in ALL_VARS:
            v_data = np.load(os.path.join(TEST_BASE, f'{v}.npy'), mmap_mode='r')[i]
            if v_data.ndim == 3: v_data = v_data[-1] 
            aux_list.append(normalize(v_data.astype(np.float32), v))
        
        aux_t = torch.from_numpy(np.stack(aux_list, axis=0)).unsqueeze(0).to(device_inf)

        with torch.no_grad():
            pred = model(pm_t, aux_t, pos_enc).numpy()
            all_preds.append(pred)

    final = np.concatenate(all_preds, axis=0)
    final = denormalize(final)
    final = np.clip(final, 0, 999)[..., np.newaxis] if final.ndim == 3 else np.clip(final, 0, 999).transpose(0, 2, 3, 1)
    
    np.save('preds.npy', final.astype(np.float32))
    print(f"✅ SUCCESS! Final shape: {final.shape}. Prodigy Predictions Saved.")
