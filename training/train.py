import torch
from torchvision.transforms import transforms as T
from torch.utils.data import DataLoader, Subset
from training.dataset import KITTIDataset
from training.model import get_model
import training.config as cfg
import os
import glob

def get_transform():
    return T.Compose([T.ToTensor()])

def collate_fn(batch):
    return tuple(zip(*batch))

def get_latest_checkpoint(checkpoint_dir):
    if not os.path.exists(checkpoint_dir): return None
    files = glob.glob(os.path.join(checkpoint_dir, "*.pth"))
    if not files: return None
    return max(files, key=os.path.getmtime) # Get the latest modified file

def train(resume=False):
    os.makedirs(cfg.CHECKPOINT_DIR, exist_ok=True)
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    print(f"Using device: {device}")

    dataset = KITTIDataset(split='training', transforms=get_transform())
    dataset_size = len(dataset)
    train_size = int(0.8 * dataset_size)
    indices = torch.randperm(dataset_size).tolist()

    train_dataset = Subset(dataset, indices[:train_size])
    train_loader = DataLoader(train_dataset, batch_size=cfg.BATCH_SIZE, shuffle=True, collate_fn=collate_fn, num_workers=2, pin_memory=True)

    model = get_model(cfg.NUM_CLASSES, device)

    params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.SGD(params, lr=cfg.LR, momentum=cfg.MOMENTUM, weight_decay=cfg.WEIGHT_DECAY)
    lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=3, gamma=0.1)

    start_epoch = 0
    if resume:
        latest_ckpt = get_latest_checkpoint(cfg.CHECKPOINT_DIR)
        if latest_ckpt:
            import logging
            print(f"Loading checkpoint: {latest_ckpt}")
            try:
                checkpoint = torch.load(latest_ckpt, map_location=device)
                model.load_state_dict(checkpoint['model_state_dict'])
                optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
                start_epoch = checkpoint['epoch'] + 1 
                print(f"Resumed from epoch {checkpoint['epoch']} | Loss: {checkpoint['loss']:.4f}")
            except Exception as e:
                print(f"Error loading checkpoint (could be bare weights): {e}")

    for epoch in range(start_epoch, cfg.EPOCHS): 
        model.train()
        epoch_loss = 0

        for i, (images, targets) in enumerate(train_loader):
            images = list(image.to(device) for image in images)
            targets = [{k: v.to(device) for k, v in t.items()} for t in targets]

            loss_dict = model(images, targets)
            losses = sum(loss for loss in loss_dict.values())

            optimizer.zero_grad()
            losses.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

            epoch_loss += losses.item()

            if i % 10 == 0:
                print(f"Epoch [{epoch}] - Batch [{i}/{len(train_loader)}] - Loss: {losses.item():.4f}")

        lr_scheduler.step()
        avg_loss = epoch_loss / len(train_loader)
        print(f"Epoch {epoch} completed! Average Loss: {avg_loss:.4f}")

        pt_path = os.path.join(cfg.CHECKPOINT_DIR, f"faster_rcnn_epoch_{epoch}.pth")
        torch.save({'epoch': epoch, 'model_state_dict': model.state_dict(), 'optimizer_state_dict': optimizer.state_dict(), 'loss': avg_loss}, pt_path)

if __name__ == "__main__":
    train(resume=True)
