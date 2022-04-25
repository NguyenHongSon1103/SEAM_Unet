hparams = dict(
    gpu = '0',
    batch_size = 12,
    epoch = 100,
    image_shape = (448, 448, 3), # H x W x 3
    train_path = dict(
        image = '/content/VOC2012/JPEGImages',
        mask = '/content/train_aff_aug_cam'
    ),
    val_path = dict(
        image = '/content/VOC2012/JPEGImages',
        mask = '/content/val_aff_cam'
    ),
    base_lr = 1e-3,
    end_lr = 3e-5,
    model_dir = '/content/drive/MyDrive/AIProject/SEAM/Unet',
    log_dir='/content/drive/MyDrive/AIProject/SEAM/Unet/logs'
)