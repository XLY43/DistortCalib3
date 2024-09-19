
from ray_diffusion.dataset.tartanair import TartanAir, AirSampler


if __name__ == "__main__":
    from torch.utils.data import Dataset, DataLoader
    from torchvision import transforms as T

    data = TartanAir('/ocean/projects/cis240055p/liuyuex/gemma/datasets/tartanair_2', catalog_path="/ocean/projects/cis240055p/liuyuex/gemma/datasets/tartanair_2/.cache/tartanair-sequences.pbz2", scale=1, augment=True)
    sampler = AirSampler(data, batch_size=4, shuffle=True)
    loader = DataLoader(data, batch_sampler=sampler, num_workers=0, pin_memory=True)

    # test_data = TartanAirTest('/data/datasets/tartanair_test', scale=1, augment=True)
    # test_sampler = AirSampler(test_data, batch_size=4, shuffle=True)
    # test_loader = DataLoader(test_data, batch_sampler=test_sampler, num_workers=4, pin_memory=True)

    for i, (image, pose, K, env_seq) in enumerate(loader):
        print(i, image.shape, pose.shape, K.shape, env_seq)
