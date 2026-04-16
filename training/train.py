import torch
from torchvision.transforms import transforms as T
from torch.utils.data import DataLoader, Subset
from training.dataset import KITTIDataset, get_transform
from training.model import get_model
import training.config as cfg
import os
import glob
from tqdm import tqdm

def collate_fn(batch):
    return tuple(zip(*batch))

def get_latest_checkpoint(checkpoint_dir):
    if not os.path.exists(checkpoint_dir): return None
    files = glob.glob(os.path.join(checkpoint_dir, "*.pth"))
    if not files: return None
    return max(files, key=os.path.getmtime)

def evaluate(model, data_loader, device):
    model.train() 
    
    for module in model.modules():
        if isinstance(module, torch.nn.BatchNorm2d):
            module.eval()

    eval_loss = 0
    with torch.no_grad():
        for images, targets in tqdm(data_loader, desc="Evaluating"):
            images = list(image.to(device) for image in images)
            targets = [{k: v.to(device) for k, v in t.items()} for t in targets]

            loss_dict = model(images, targets)
            losses = sum(loss for loss in loss_dict.values())
            eval_loss += losses.item()

    return eval_loss / len(data_loader)

def train(resume=False):
    os.makedirs(cfg.CHECKPOINT_DIR, exist_ok=True)
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    print(f"Using device: {device}")

    full_dataset = KITTIDataset(split='training', transforms=None)
    dataset_size = len(full_dataset)
    train_size = int(0.8 * dataset_size)
    indices = torch.randperm(dataset_size).tolist()

    train_indices = indices[:train_size]
    val_indices   = indices[train_size:]

    train_base = KITTIDataset(split='training', transforms=get_transform(train=True))
    val_base   = KITTIDataset(split='training', transforms=get_transform(train=False))

    train_dataset = Subset(train_base, train_indices)
    val_dataset   = Subset(val_base,   val_indices)

    train_loader = DataLoader(train_dataset, batch_size=cfg.BATCH_SIZE, shuffle=True, 
                             collate_fn=collate_fn, num_workers=2, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=cfg.BATCH_SIZE, shuffle=False, 
                           collate_fn=collate_fn, num_workers=2, pin_memory=True)

    print(f"Train samples: {len(train_dataset)}, Val samples: {len(val_dataset)}")

    model = get_model(cfg.NUM_CLASSES, device)

    params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.SGD(params, lr=cfg.LR, momentum=cfg.MOMENTUM, weight_decay=cfg.WEIGHT_DECAY)
    
    # Warmup scheduler for first epoch
    warmup_scheduler = torch.optim.lr_scheduler.LinearLR(
        optimizer, start_factor=0.1, total_iters=cfg.WARMUP_EPOCHS
    )
    # Main scheduler
    lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=cfg.LR_STEP_SIZE, gamma=cfg.LR_GAMMA)

    start_epoch = 0
    best_val_loss = float('inf')

    if resume:
        latest_ckpt = get_latest_checkpoint(cfg.CHECKPOINT_DIR)
        if latest_ckpt:
            print(f"Loading checkpoint: {latest_ckpt}")
            try:
                checkpoint = torch.load(latest_ckpt, map_location=device)
                model.load_state_dict(checkpoint['model_state_dict'])
                optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
                start_epoch = checkpoint['epoch'] + 1
                best_val_loss = checkpoint.get('best_val_loss', float('inf'))
                print(f"Resumed from epoch {checkpoint['epoch']} | Train Loss: {checkpoint['loss']:.4f} | Val Loss: {checkpoint.get('val_loss', 'N/A')}")
            except Exception as e:
                print(f"Error loading checkpoint (will start fresh): {e}")
        else:
            print("No checkpoint found, starting fresh training")

    for epoch in range(start_epoch, cfg.EPOCHS):
        model.train()
        epoch_loss = 0

        pbar = tqdm(train_loader, desc=f"Epoch {epoch}/{cfg.EPOCHS}")
        for i, (images, targets) in enumerate(pbar):
            images = list(image.to(device) for image in images)
            targets = [{k: v.to(device) for k, v in t.items()} for t in targets]

            loss_dict = model(images, targets)
            losses = sum(loss for loss in loss_dict.values())

            optimizer.zero_grad()
            losses.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

            epoch_loss += losses.item()
            pbar.set_postfix({'loss': f'{losses.item():.4f}', 'avg_loss': f'{epoch_loss/(i+1):.4f}'})

        # Warmup for first few epochs
        if epoch < cfg.WARMUP_EPOCHS:
            warmup_scheduler.step()
        else:
            lr_scheduler.step()

        avg_train_loss = epoch_loss / len(train_loader)
        print(f"Epoch {epoch} - Train Loss: {avg_train_loss:.4f}")

        # Validation
        val_loss = evaluate(model, val_loader, device)
        print(f"Epoch {epoch} - Val Loss: {val_loss:.4f}")

        # Save best model
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_path = os.path.join(cfg.CHECKPOINT_DIR, "best_model.pth")
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'loss': avg_train_loss,
                'val_loss': val_loss,
                'best_val_loss': best_val_loss
            }, best_path)
            print(f"Saved best model with val loss: {best_val_loss:.4f}")

        # Save checkpoint every epoch
        pt_path = os.path.join(cfg.CHECKPOINT_DIR, f"faster_rcnn_epoch_{epoch}.pth")
        torch.save({
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'loss': avg_train_loss,
            'val_loss': val_loss,
            'best_val_loss': best_val_loss
        }, pt_path)

        print(f"Current LR: {optimizer.param_groups[0]['lr']:.6f}")

    print(f"Training completed! Best validation loss: {best_val_loss:.4f}")

if __name__ == "__main__":
    train(resume=False)
