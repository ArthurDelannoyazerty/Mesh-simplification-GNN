import torch

from torch.utils.tensorboard import SummaryWriter
from train import train


# writer = SummaryWriter()


with torch.profiler.profile(
            schedule=torch.profiler.schedule(wait=0, warmup=0, active=1, repeat=0),
            on_trace_ready=torch.profiler.tensorboard_trace_handler('runs/'),
            record_shapes=False,
            profile_memory=False,
            with_stack=False
        ) as prof:
  
    prof.step()
    train()