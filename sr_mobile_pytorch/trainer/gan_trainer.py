import os
import torch
from torch.nn import L1Loss
from torch.utils.data import DataLoader
from torch.optim import Adam
from torch.optim.lr_scheduler import StepLR
from tqdm.auto import tqdm
import wandb

from sr_mobile_pytorch.model import AnchorBasedPlainNet, DCGANDiscriminator
from sr_mobile_pytorch.trainer.metrics import calculate_psnr
from sr_mobile_pytorch.trainer.utils import seed_everything, logger
from sr_mobile_pytorch.trainer.losses import ContentLossResNetSimCLR, GANLoss


class GANTrainer:
    def __init__(self, model_args, training_args, train_dataset, test_dataset):
        seed_everything(training_args["seed"])
        self.model_args = model_args
        self.training_args = training_args

        self.train_loader = DataLoader(
            dataset=train_dataset,
            batch_size=training_args["train_batch_size"],
            shuffle=True,
            num_workers=training_args["num_workers"],
        )

        self.test_loader = DataLoader(
            dataset=test_dataset,
            batch_size=1,
            shuffle=False,
            num_workers=training_args["num_workers"],
        )

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.generator = AnchorBasedPlainNet(**model_args)
        self.generator.load_state_dict(
            torch.load(training_args["generator_weights"]), strict=True
        )
        self.generator = self.generator.to(self.device)

        self.discriminator = DCGANDiscriminator(features_d=4).to(self.device)

        self.pixelwise_loss = L1Loss()
        self.content_loss = ContentLossResNetSimCLR(
            training_args["resnet_weights"], self.device
        )
        self.gan_loss = GANLoss()

        self.opt_d = Adam(
            self.discriminator.parameters(),
            lr=training_args["discriminator_learning_rate"],
        )
        self.opt_g = Adam(
            self.generator.parameters(), lr=training_args["generator_learning_rate"]
        )

        num_training_steps = len(self.train_loader) * training_args["epochs"]
        step_size = num_training_steps // 2
        self.scheduler_d = StepLR(self.opt_d, step_size=step_size, gamma=0.1)
        self.scheduler_g = StepLR(self.opt_g, step_size=step_size, gamma=0.1)

        self.state = {"best_psnr": None, "best_loss": None}
        wandb.watch(self.generator)
        wandb.watch(self.discriminator)

    def fit(self):
        for epoch in range(self.training_args["epochs"]):
            epoch_generator_loss, epoch_discriminator_loss = 0.0, 0.0
            epoch_content_loss, epoch_pixelwise_loss = 0.0, 0.0
            epoch_perceptual_loss = 0.0
            self.generator.train()
            self.discriminator.train()

            for lr, hr in tqdm(self.train_loader, total=len(self.train_loader)):
                lr, hr = lr.to(self.device), hr.to(self.device)
                sr = self.generator(lr)

                # train discriminator
                self.opt_d.zero_grad()

                hr_out = self.discriminator(hr)
                sr_out = self.discriminator(sr.detach())

                discriminator_loss = self.gan_loss.discriminator_loss(hr_out, sr_out)
                discriminator_loss.backward()
                self.opt_d.step()
                self.scheduler_d.step()

                # train generator
                self.opt_g.zero_grad()

                sr_out = self.discriminator(sr)
                generator_loss = self.gan_loss.generator_loss(sr_out)
                content_loss = self.content_loss(hr, sr)
                pixelwise_loss = self.pixelwise_loss(hr, sr)
                perceptual_loss = (
                    content_loss + 0.1 * generator_loss + 0.1 * pixelwise_loss
                )

                perceptual_loss.backward()
                self.opt_g.step()
                self.scheduler_g.step()

                epoch_generator_loss += generator_loss.item()
                epoch_discriminator_loss += discriminator_loss.item()
                epoch_content_loss += content_loss.item()
                epoch_pixelwise_loss += pixelwise_loss.item()
                epoch_perceptual_loss += perceptual_loss.item()

            train_generator_loss = epoch_discriminator_loss / len(self.train_loader)
            train_discriminator_loss = epoch_discriminator_loss / len(self.train_loader)
            train_content_loss = epoch_content_loss / len(self.train_loader)
            train_pixelwise_loss = epoch_pixelwise_loss / len(self.train_loader)
            train_perceptual_loss = epoch_perceptual_loss / len(self.train_loader)
            test_loss, test_psnr = self.evaluate()

            self.save_best_model(test_loss, test_psnr)
            self.report_results(
                train_generator_loss,
                train_discriminator_loss,
                train_content_loss,
                train_pixelwise_loss,
                train_perceptual_loss,
                test_loss,
                test_psnr,
                epoch + 1,
            )

            logger.info(
                f"Epoch: {epoch:4} | Train Generator Loss: {train_generator_loss:.4f} | Train Discriminator Loss: {train_discriminator_loss:.4f} | Train Content Loss: {train_content_loss:.4f} | Train Pixelwise Loss: {train_pixelwise_loss:.4f} | Train Perceptual Loss: {train_perceptual_loss:.4f} | Test Pixelwise Loss: {test_loss:.4f} | PSNR: {test_psnr:.2f}"
            )

    def evaluate(self):
        total_loss, total_psnr = 0.0, 0.0
        self.generator.eval()

        with torch.no_grad():
            for lr, hr in tqdm(self.test_loader, total=len(self.test_loader)):
                lr, hr = lr.to(self.device), hr.to(self.device)

                output = self.generator(lr)

                loss = self.pixelwise_loss(output, hr)
                total_psnr += calculate_psnr(
                    output.cpu().detach().numpy(), hr.cpu().detach().numpy()
                )
                total_loss += loss.item()

            test_loss = total_loss / len(self.test_loader)
            test_psnr = total_psnr / len(self.test_loader)

            return test_loss, test_psnr

    def report_results(
        self,
        train_generator_loss,
        train_discriminator_loss,
        train_content_loss,
        train_pixelwise_loss,
        train_perceptual_loss,
        test_loss,
        test_psnr,
        step,
    ):
        wandb.log(
            {
                "train-generator-loss": train_generator_loss,
                "train-discriminator-loss": train_discriminator_loss,
                "train-content-loss": train_content_loss,
                "train-pixelwise-loss": train_pixelwise_loss,
                "train-perceptual-loss": train_perceptual_loss,
                "test-pixelwise-loss": test_loss,
                "test-psnr": test_psnr,
            },
            step=step,
        )

    def save_best_model(self, current_loss, current_psnr):
        save_path = f"{self.training_args['outdir']}/gan"
        os.makedirs(save_path, exist_ok=True)

        torch.save(self.generator.state_dict(), f"{save_path}/generator.pth")
        torch.save(self.discriminator.state_dict(), f"{save_path}/discriminator.pth")

        if self.state["best_loss"] == None or current_loss < self.state["best_loss"]:
            self.state["best_loss"] = current_loss
            torch.save(
                self.generator.state_dict(), f"{save_path}/generator_best_loss.pth"
            )
            torch.save(
                self.discriminator.state_dict(),
                f"{save_path}/discriminator_best_loss.pth",
            )

        if self.state["best_psnr"] == None or current_psnr > self.state["best_psnr"]:
            self.state["best_psnr"] = current_psnr
            torch.save(
                self.generator.state_dict(), f"{save_path}/generator_best_psnr.pth"
            )
            torch.save(
                self.discriminator.state_dict(),
                f"{save_path}/discriminator_best_psnr.pth",
            )
