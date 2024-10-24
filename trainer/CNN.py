import os 

from pytorch_lightning import LightningModule, Trainer
from pytorch_lightning.trainer import seed_everything
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping
from pytorch_lightning.loggers import TensorBoardLogger
from random import randint
import shutil
import torch
import torch.nn as nn
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torchinfo import summary
from torchvision.transforms import v2

from data.MNISTDataset import MNISTDataModule 
from model.CNN import CNN
from model.loss import MyCrossEntrpyLoss
from util.get_optimizer import get_optimizer


class LitModel(LightningModule):
    def __init__(self, criterion=nn.CrossEntropyLoss(), optimizer=torch.optim.Adam, learning_rate: float = 0.001, momentum: float = 0.9, weight_decay: float = 0.0001):
        super().__init__()
        
        self.model = CNN()
        
        self.example_input_array = torch.randn(1, 1, 28, 28)

        self.criterion = criterion
        self.optimizer = optimizer
        self.learning_rate = learning_rate
        self.momentum = momentum
        self.weight_decay = weight_decay
    
        # initialize the model
        # self.model.apply(self.initialize)
        
    # kaiming initialization 
    def initialize(self, m):
        if isinstance(m, nn.Linear):
            nn.init.kaiming_normal_(m.weight, mode='fan_in', nonlinearity='relu')
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.Conv2d):
            nn.init.kaiming_normal_(m.weight, mode='fan_in', nonlinearity='relu')
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.BatchNorm2d):
            nn.init.constant_(m.weight, 1)
            nn.init.constant_(m.bias, 0)
            
    def forward(self, x):
        return self.model(x)
    
    def _model_forward(self, x):
        """
            process the output of the model
        """
        x = self(x)
        
        return x
    
    def _train_metrics(self, pred, y):
        pred = torch.argmax(pred, dim=1)
        y = torch.argmax(y, dim=1)
        accuracy = torch.sum(pred == y).item() / len(y)
        self.log('train_accuracy', accuracy, on_epoch=True, on_step=False, sync_dist=True)

    def _val_metrics(self, pred, y):
        pred = torch.argmax(pred, dim=1)
        y = torch.argmax(y, dim=1)
        accuracy = torch.sum(pred == y).item() / len(y)
        self.log('val_accuracy', accuracy, on_epoch=True, on_step=False, sync_dist=True)

    def _test_metrics(self, pred, y):
        pred = torch.argmax(pred, dim=1)
        y = torch.argmax(y, dim=1)
        accuracy = torch.sum(pred == y).item() / len(y)
        self.log('test_accuracy', accuracy, on_epoch=True, on_step=False, sync_dist=True)

    def training_step(self, batch, batch_idx):
        x, y = batch
        y_pred = self._model_forward(x)
        
        loss = self.criterion(y_pred, y)
        self.log('train_loss', loss, on_epoch=True, on_step=False, sync_dist=True)
        
        self._train_metrics(y_pred, y)
        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch
        y_pred = self._model_forward(x)
        
        loss = self.criterion(y_pred, y)

        self.log('val_loss', loss, on_epoch=True, on_step=False, sync_dist=True)
        
        self._val_metrics(y_pred, y)
    
    def test_step(self, batch, batch_idx):
        x, y = batch
        y_pred = self._model_forward(x)
        
        loss = self.criterion(y_pred, y)

        self.log('test_loss', loss, on_epoch=True, on_step=False, sync_dist=True)
        
        self._test_metrics(y_pred, y)
        
    def configure_optimizers(self):
        if self.optimizer.__name__ == 'Adam' or self.optimizer.__name__ == 'AdamW':
            optimizer = self.optimizer(self.parameters(), lr=self.learning_rate, weight_decay=self.weight_decay)
        else:
            optimizer = self.optimizer(self.parameters(), lr=self.learning_rate, momentum=self.momentum, weight_decay=self.weight_decay)

        lr_scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=15)
        return [
            {
                'optimizer': optimizer,
                'lr_scheduler': {
                    'scheduler': lr_scheduler,
                    'monitor': 'val_loss',
                }
            }
        ]

def train(config):
    # Set seed
    if 'seed' in config['trainer'] and config['trainer']['seed']:
        seed = config['trainer']['seed']
    else:
        seed = randint(0, 4294967295)
    seed_everything(seed, workers=True)
    
    # Load the dataset
    dataset = MNISTDataModule(
        train_path=config['paths']['train_path'],
        val_path=config['paths']['val_path'],
        test_path=config['paths']['test_path'],
        batch_size=config['trainer']['batch_size'],
        train_shuffle=config['trainer']['train_shuffle'],
        val_shuffle=config['trainer']['val_shuffle'],
        num_workers=config['trainer']['num_workers'],
        train_transform=v2.Compose([
            v2.ToImage(),
            v2.ToDtype(torch.float32, scale=True),
            v2.RandomHorizontalFlip(),
            v2.RandomVerticalFlip(),
        ]),
        val_transform=v2.Compose([
            v2.ToImage(),
            v2.ToDtype(torch.float32, scale=True),
        ])
    )

    # Get the optimizer
    optimizer = get_optimizer(config['trainer']['optimizer'])
    # Get the criterion
    criterion = MyCrossEntrpyLoss()

    # Load the model
    lit_model = LitModel(
        criterion=criterion,
        optimizer=optimizer,
        learning_rate=config['trainer']['learning_rate'],
        momentum=config['trainer']['momentum'],
        weight_decay=config['trainer']['weight_decay'],
    )    

    # Set the precision of the matrix multiplication to float32
    torch.set_float32_matmul_precision(config['trainer']['precision'])

    # Add TensorBoard logger
    logger = TensorBoardLogger(save_dir=config['logger']['log_dir'], name=config['logger']['group'], version=config['trainer']['version'])
    logger.log_hyperparams(config)
    logger.log_hyperparams({'seed': seed})
    logger.experiment.add_text('Note', config['logger']['description'])
    # logger.experiment.add_graph(lit_model.model, lit_model.example_input_array)   # visualizing the model in TensorBoard
    
    # Save the model summary
    tmp = summary(lit_model.model, input_size=lit_model.example_input_array.shape, col_names=("input_size", "output_size", "num_params", "kernel_size", "mult_adds"), verbose=1)
    with open(f"{config['logger']['log_dir']}/{config['logger']['group']}/version_{config['trainer']['version']}/model_summary.txt", 'w') as f:
        f.write(str(tmp))
    del tmp
    
     # Save the current script
    current_script_path = os.path.abspath(__file__)
    destination_folder = f"{config['logger']['log_dir']}/{config['logger']['group']}/version_{config['trainer']['version']}"
    os.makedirs(destination_folder, exist_ok=True)
    destination_file = os.path.join(destination_folder, os.path.basename(current_script_path))
    shutil.copy2(current_script_path, destination_file)

    # Create a trainer
    trainer = Trainer(
        accelerator=config['trainer']['accelerator'],
        devices=config['trainer']['devices'],
        max_epochs=config['trainer']['max_epochs'],
        max_steps=config['trainer']['max_steps'],
        logger=logger,
        callbacks=[
            ModelCheckpoint(filename='{epoch}-{val_loss:.4f}-{val_accuracy:.4f}', every_n_epochs=1, monitor='val_loss', save_top_k=5, mode='min', save_last=True),
            EarlyStopping(monitor='val_loss', patience=50, mode='min', verbose=True),
        ],
        # strategy='ddp_find_unused_parameters_true',   # For debugging, find unused parameters
        fast_dev_run=config['fast_dev_run']  # For checking the correctness of the model
    )

    # train and validation
    if 'ckpt_path' in config['paths'] and config['paths']['ckpt_path']:
        trainer.fit(model=lit_model, datamodule=dataset, ckpt_path=config['paths']['ckpt_path'])
    else:
        trainer.fit(model=lit_model, datamodule=dataset)
    
    # test
    trainer.test(model=lit_model, datamodule=dataset, ckpt_path="best", verbose=True)
    
def predict(config):
    lit_model = LitModel.load_from_checkpoint(config['paths']['ckpt_path'])
    lit_model.eval()
    