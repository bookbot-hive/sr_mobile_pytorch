import os
import torch
import torch.nn.functional as F
from torch.nn import L1Loss
from torch.utils.data import DataLoader
from torch.optim import Adam
from tqdm.auto import tqdm
import wandb

from sr_mobile_pytorch.model import AnchorBasedPlainNet
from sr_mobile_pytorch.trainer.schedulers import get_linear_schedule_with_warmup
from sr_mobile_pytorch.trainer.metrics import calculate_psnr
from sr_mobile_pytorch.trainer.utils import seed_everything, logger


class Trainer:
    def __init__(self, model_args, training_args, train_dataset, test_dataset):
        self.model_args = model_args
        self.training_args = training_args
        seed_everything(training_args["seed"])

        self.train_loader = DataLoader(
            train_dataset,
            training_args["train_batch_size"],
            shuffle=True,
            num_workers=8,
        )

        self.test_loader = DataLoader(
            test_dataset,
            training_args["test_batch_size"],
            collate_fn=self.collate_fn,
            shuffle=False,
            num_workers=8,
        )

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = AnchorBasedPlainNet(**model_args)
        self.model = self.model.to(self.device)

        self.criterion = L1Loss()
        self.optimizer = Adam(
            self.model.parameters(),
            training_args["learning_rate"],
            weight_decay=training_args["weight_decay"],
        )

        num_training_steps = len(self.train_loader) * training_args["epochs"]
        warmup_steps = int(training_args["warmup_ratio"] * num_training_steps)
        self.scheduler = get_linear_schedule_with_warmup(
            self.optimizer, warmup_steps, num_training_steps, last_epoch=-1
        )

        self.state = {"best_psnr": None, "best_loss": None}
        wandb.watch(self.model)

    def fit(self):
        for epoch in range(self.training_args["epochs"]):
            epoch_loss = 0.0
            self.model.train()

            for lr, hr in tqdm(self.train_loader, total=len(self.train_loader)):
                lr, hr = lr.to(self.device), hr.to(self.device)

                self.optimizer.zero_grad()
                output = self.model(lr)

                loss = self.criterion(output, hr)
                epoch_loss += loss.item()
                loss.backward()

                self.optimizer.step()
                self.scheduler.step()

            train_loss = epoch_loss / len(self.train_loader)
            test_loss, test_psnr = self.evaluate()

            self.save_best_model(test_loss, test_psnr)
            self.report_results(train_loss, test_loss, test_psnr, epoch + 1)

            logger.info(
                "Epoch: {epoch:4} | Train Loss: {train_loss:.4f} | Test Loss: {test_loss:.4f} | PSNR: {test_psnr:.2f}"
            )

    def evaluate(self):
        total_loss, total_psnr = 0.0, 0.0
        self.model.eval()

        with torch.no_grad():
            for lr, hr in tqdm(self.test_loader, total=len(self.test_loader)):
                lr, hr = lr.to(self.device), hr.to(self.device)

                output = self.model(lr)

                loss = self.criterion(output, hr)
                total_psnr += calculate_psnr(
                    output.cpu().detach().numpy(), hr.cpu().detach().numpy()
                )
                total_loss += loss.item()

            test_loss = total_loss / len(self.test_loader)
            test_psnr = total_psnr / len(self.test_loader)

            return test_loss, test_psnr

    def collate_fn(self, batch):
        lr, hr, = zip(*batch)

        max_lr_shape = max([t.shape for t in lr])
        max_hr_shape = max([t.shape for t in hr])

        pad_fn = lambda input, new_shape: F.pad(
            input=input,
            pad=(
                0,
                new_shape[2] - input.shape[2],
                0,
                new_shape[1] - input.shape[1],
                0,
                0,
            ),
        )

        lr_padded = torch.stack([pad_fn(torch.tensor(t), max_lr_shape) for t in lr])
        hr_padded = torch.stack([pad_fn(torch.tensor(t), max_hr_shape) for t in hr])

        return lr_padded, hr_padded

    def report_results(self, train_loss, test_loss, test_psnr, step):
        wandb.log(
            {"train-loss": train_loss, "test-loss": test_loss, "test-psnr": test_psnr},
            step=step,
        )

    def save_best_model(self, current_loss, current_psnr):
        save_path = f"{self.training_args['outdir']}/weights"
        os.makedirs(save_path, exist_ok=True)

        torch.save(self.model.state_dict(), f"{save_path}/model.pth")

        if self.state["best_loss"] == None or current_loss < self.state["best_loss"]:
            self.state["best_loss"] = current_loss
            torch.save(self.model.state_dict(), f"{save_path}/best_loss.pth")

        if self.state["best_psnr"] == None or current_psnr > self.state["best_psnr"]:
            self.state["best_psnr"] = current_psnr
            torch.save(self.model.state_dict(), f"{save_path}/best_psnr.pth")

