import torch
from lightning.pytorch.strategies import DDPStrategy

class CustomDDPStrategy(DDPStrategy):
    def setup_environment(self):
        # Ensure that the process group is initialized using the gloo backend
        if not torch.distributed.is_initialized():
            torch.distributed.init_process_group(
                backend='gloo',
                rank=self.global_rank,
                world_size=self.world_size
            )
        super().setup_environment()

    def configure_ddp(self):
        # Set up DistributedDataParallel to use only the local_rank for device IDs
        self.model = torch.nn.parallel.DistributedDataParallel(
            self.model,
            device_ids=[self.local_rank], 
            output_device=self.local_rank,
        )
