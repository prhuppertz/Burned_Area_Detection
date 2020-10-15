from imgaug import augmenters as iaa

seq = iaa.Sequential(
    [
        iaa.AddToHueAndSaturation((-20, 20)),
        iaa.Multiply((0.5, 1.2), per_channel=0.2),
        iaa.Sometimes(0.3, iaa.Add((-40, 40), per_channel=0.5)),
        iaa.Fliplr(0.5),
        iaa.Flipud(0.5),
    ]
)
