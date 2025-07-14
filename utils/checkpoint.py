import os
import torch

class CheckpointManager:
    def __init__(self, save_dir):
        self.save_dir = save_dir
        self.ckpt_path = os.path.join(save_dir, "last.ckpt")

    def has_checkpoint(self):
        return os.path.exists(self.ckpt_path)

    def save_checkpoint(self, model, optimizer, scheduler, epoch, loss):
        torch.save({
            'model_state': model.state_dict(),
            'optimizer_state': optimizer.state_dict(),
            'scheduler_state': scheduler.state_dict(),
            'epoch': epoch,
            'loss': loss
        }, self.ckpt_path)

    def load_checkpoint(self, model, optimizer, scheduler):
        checkpoint = torch.load(self.ckpt_path)
        model.load_state_dict(checkpoint['model_state'])
        optimizer.load_state_dict(checkpoint['optimizer_state'])
        scheduler.load_state_dict(checkpoint['scheduler_state'])
        return checkpoint
