import argparse
from pathlib import Path

import monai
import tomlkit
import torch
from matplotlib import pyplot as plt
from monai.networks.nets import DiffusionModelUNet
from monai.networks.schedulers import DDPMScheduler
from tqdm import tqdm

from train import val_dataset


def main(cfg_path: str, num_workers: int, device: str = 'cuda'):
    cfg_path = Path(cfg_path)
    cfg = tomlkit.loads(cfg_path.read_text('utf-8'))

    root = Path(cfg['train']['root'])
    if not root.is_absolute():
        root = cfg_path.parent / root
    root = root.resolve()

    root.mkdir(parents=True, exist_ok=True)

    final_pth = root / 'checkpoints' / 'final.pth'
    ckpt = torch.load(final_pth, map_location=device)

    batch_size = cfg['train']['batch_size']
    spatial_size = ckpt['dataset']['spatial_size']
    num_timesteps = ckpt['scheduler']['num_train_timesteps']

    noise = lambda: torch.randn((1, 1, *spatial_size)).to(device)  # 高斯随机纯噪声，Batch/Channel/DHW

    model = DiffusionModelUNet(**ckpt['model']).to(device).eval()
    model.load_state_dict(ckpt['state']['model'])

    scheduler = DDPMScheduler(num_train_timesteps=ckpt['scheduler']['num_train_timesteps'])
    scheduler.set_timesteps(num_inference_steps=num_timesteps)  # 从纯噪声开始去噪

    interval = 100  # 保留去噪中间图
    intermediates = []

    if INPAINTING := True:
        val_loader = val_dataset(root.as_posix(), batch_size, num_workers)
        val_batch = monai.utils.first(val_loader)['image']
        val_image = val_batch[1:2]

        mask = torch.ones_like(val_image)
        mask[:, :, 20:40, 30:80] = 0
        val_image_masked = val_image * mask

        plt.subplot(1, 3, 1)
        plt.imshow(val_image[0, 0, ...], cmap="gray")
        plt.title("Original image")
        plt.axis("off")
        plt.subplot(1, 3, 2)
        plt.imshow(mask[0, 0, ...], cmap="gray")
        plt.axis("off")
        plt.title("Mask")
        plt.subplot(1, 3, 3)
        plt.imshow(val_image_masked[0, 0, ...], cmap="gray")
        plt.axis("off")
        plt.title("Masked image")
        plt.show()

        mask = mask.to(device)
        val_image_masked = val_image_masked.to(device)

        inpainted = noise()

        with torch.no_grad(), torch.amp.autocast(device_type=device, enabled=True):
            for t in tqdm(scheduler.timesteps):  # 去噪
                for u in range(max(1, num_resample_steps := 4)):  # 接缝修正

                    # 已知区域从 0 加噪到 t-1
                    if t > 0:
                        val_image_inpainted_prev_known = scheduler.add_noise(
                            original_samples=val_image_masked,
                            noise=noise(),
                            timesteps=torch.Tensor((t - 1,)).to(device).long()
                        )
                    else:
                        val_image_inpainted_prev_known = val_image_masked

                    # 未知区域从 t 去噪到 t-1
                    if t > 0:
                        model_output = model(inpainted, timesteps=torch.Tensor((t,)).to(device).long())
                        val_image_inpainted_prev_unknown, _ = scheduler.step(model_output, t, inpainted)

                    # 拼接
                    inpainted = torch.where(
                        mask == 1,
                        val_image_inpainted_prev_known,  # 已知区域
                        val_image_inpainted_prev_unknown  # 网络生成的未知区域
                    )

                    # 拼接图从 t-1 加噪到 t，接缝修正
                    if t > 0 and u < num_resample_steps - 1:
                        inpainted *= torch.sqrt(1 - scheduler.betas[t - 1])
                        inpainted += noise() * torch.sqrt(scheduler.betas[t - 1])

                if interval > 0 and t % interval == 0:
                    intermediates.append(inpainted.detach().cpu())


    else:
        with torch.no_grad(), torch.amp.autocast(device_type=device, enabled=True):  # 混合精度
            image = noise()

            for t in tqdm(scheduler.timesteps):
                model_output = model(image, timesteps=torch.Tensor((t,)).to(device).long())
                image, _ = scheduler.step(model_output, t, image)

                if interval > 0 and t % interval == 0:
                    intermediates.append(image.detach().cpu())

            # infer = DiffusionInferer(scheduler)
            # image = infer.sample(
            #     input_noise=image,
            #     diffusion_model=model, scheduler=scheduler,
            #     save_intermediates=interval > 0, intermediate_steps=interval,
            # )
            # if interval > 0:
            #     image, intermediates = image

    if len(intermediates):
        chain = torch.cat(intermediates, dim=-1)

        plt.imshow(chain[0, 0], vmin=0, vmax=1, cmap='gray')
        plt.tight_layout()
        plt.axis('off')
        plt.show()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', required=True)
    parser.add_argument('--num_workers', type=int, default=16)
    args = parser.parse_args()

    # monai.config.print_config()
    assert torch.cuda.is_available()

    try:
        main(args.config, args.num_workers)
    except KeyboardInterrupt:
        print('Keyboard interrupted')
