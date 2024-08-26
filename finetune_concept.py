import torch
from torch.utils.data import Dataset, DataLoader
from torch.optim import AdamW
from PIL import Image
import os
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint
from ldm.util import instantiate_from_config
from ldm.models.diffusion.ddpm import LatentDiffusion

# Custom Dataset to load image-caption pairs
class CustomDataset(Dataset):
    def __init__(self, image_dir, captions_file, transform=None):
        self.image_dir = image_dir
        self.transform = transform

        # Load captions with additional debug information
        self.captions = {}
        with open(captions_file, 'r') as f:
            lines = f.readlines()

        for i, line in enumerate(lines):
            parts = line.strip().split('\t')
            if len(parts) == 2:  # Ensure there are exactly two parts
                filename, caption = parts
                self.captions[filename] = caption
            else:
                print(f"Skipping line {i+1} due to incorrect format: {line.strip()}")  # Debugging output

        # List image filenames
        self.image_filenames = list(self.captions.keys())
        
        
        # # Load captions
        # with open(captions_file, 'r') as f:
        #     lines = f.readlines()
        # self.captions = {line.split('\t')[0]: line.split('\t')[1].strip() for line in lines}

        # # List image filenames
        # self.image_filenames = list(self.captions.keys())

    def __len__(self):
        return len(self.image_filenames)

    def __getitem__(self, idx):
        img_name = self.image_filenames[idx]
        img_path = os.path.join(self.image_dir, img_name)
        image = Image.open(img_path).convert('RGB')

        if self.transform:
            image = self.transform(image)

        caption = self.captions[img_name]
        return {'image': image, 'caption': caption}

# Lightning Module for LoRA Finetuning
class LoraFineTuneModel(pl.LightningModule):
    def __init__(self, model_config, learning_rate, pretrained_model_path=None):
        super().__init__()
        self.model = instantiate_from_config(model_config)
        self.learning_rate = learning_rate

        # Load pretrained model weights if provided
        if pretrained_model_path:
            self.model.load_state_dict(torch.load(pretrained_model_path, map_location=self.device))
            print(f"Loaded pretrained model from {pretrained_model_path}")

        # Freeze all model parameters except LoRA layers
        for param in self.model.parameters():
            param.requires_grad = False
        for name, param in self.model.named_parameters():
            if 'lora_layer.A' in name or 'lora_layer.B' in name:
                param.requires_grad = True

    def forward(self, x, c):
        return self.model(x, c)

    def training_step(self, batch, batch_idx):
        x, c = batch['image'], batch['caption']
        loss = self.model(x, c)['loss']
        self.log('train_loss', loss)
        return loss

    def configure_optimizers(self):
        # Only optimize the parameters in LoRA layers
        lora_params = [param for name, param in self.model.named_parameters() if param.requires_grad]
        optimizer = AdamW(lora_params, lr=self.learning_rate)
        return optimizer

def main():
    # Load configuration
    model_config = {
        'target': 'ldm.models.diffusion.ddpm.LatentDiffusion',
        'params': {
            'linear_start': 0.00085,
            'linear_end': 0.012,
            'num_timesteps_cond': 1,
            'log_every_t': 200,
            'timesteps': 1000,
            'first_stage_key': 'image',
            'cond_stage_key': 'caption',
            'image_size': 32,
            'channels': 4,
            'cond_stage_trainable': True,
            'conditioning_key': 'crossattn',
            'monitor': 'val/loss_simple_ema',
            'scale_factor': 0.18215,
            'use_ema': False,
            'unet_config': {
                'target': 'ldm.modules.diffusionmodules.openaimodel.UNetModel',
                'params': {
                    'image_size': 32,
                    'in_channels': 4,
                    'out_channels': 4,
                    'model_channels': 320,
                    'attention_resolutions': [4, 2, 1],
                    'num_res_blocks': 2,
                    'channel_mult': [1, 2, 4, 4],
                    'num_heads': 8,
                    'use_spatial_transformer': True,
                    'transformer_depth': 1,
                    'context_dim': 1280,
                    'use_checkpoint': True,
                    'legacy': False,
                },
            },
            'first_stage_config': {
                'target': 'ldm.models.autoencoder.AutoencoderKL',
                'params': {
                    'embed_dim': 4,
                    'monitor': 'val/rec_loss',
                    'ddconfig': {
                        'double_z': True,
                        'z_channels': 4,
                        'resolution': 256,
                        'in_channels': 3,
                        'out_ch': 3,
                        'ch': 128,
                        'ch_mult': [1, 2, 4, 4],
                        'num_res_blocks': 2,
                        'attn_resolutions': [],
                        'dropout': 0.0,
                    },
                    'lossconfig': {
                        'target': 'torch.nn.Identity',
                    },
                },
            },
            'cond_stage_config': {
                'target': 'ldm.modules.encoders.modules.BERTEmbedder',
                'params': {
                    'n_embed': 1280,
                    'n_layer': 32,
                },
            },
        },
    }

    # Initialize dataset and dataloader
    dataset = CustomDataset(image_dir='benchmark_dataset/person_1', captions_file='benchmark_dataset/person_1/captions.txt')
    dataloader = DataLoader(dataset, batch_size=32, shuffle=True, num_workers=4)

    # Instantiate model with the path to the pretrained model
    model = LoraFineTuneModel(model_config, learning_rate=5e-5, pretrained_model_path='models/ldm/text2img-large/model.ckpt')

    # Setup ModelCheckpoint callback
    checkpoint_callback = ModelCheckpoint(
        dirpath='./checkpoints',
        filename='finetuned-model-{epoch:02d}-{train_loss:.2f}',
        save_top_k=3,
        monitor='train_loss',
        mode='min'
    )

    # Setup trainer with ModelCheckpoint callback
    trainer = pl.Trainer(
        max_epochs=10,
        gpus=1,
        precision=16,
        callbacks=[checkpoint_callback],
    )

    # Start training
    trainer.fit(model, train_dataloaders=dataloader)

if __name__ == '__main__':
    main()
