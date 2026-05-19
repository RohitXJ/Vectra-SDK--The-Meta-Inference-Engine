import torch.nn as nn
import torchvision.models as models

def get_encoder(backbone_name, image_format):
    if backbone_name == 'resnet18':
        model = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)
        if image_format == "L":
            model.conv1 = nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False)
        model = nn.Sequential(*list(model.children())[:-1], nn.Flatten())

    elif backbone_name == 'resnet34':
        model = models.resnet34(weights=models.ResNet34_Weights.IMAGENET1K_V1)
        if image_format == "L":
            model.conv1 = nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False)
        model = nn.Sequential(*list(model.children())[:-1], nn.Flatten())

    elif backbone_name == 'resnet50':
        model = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V1)
        if image_format == "L":
            model.conv1 = nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False)
        model = nn.Sequential(*list(model.children())[:-1], nn.Flatten())

    elif backbone_name == 'mobilenet_v2':
        model = models.mobilenet_v2(weights=models.MobileNet_V2_Weights.IMAGENET1K_V1)
        if image_format == "L":
            model.features[0][0] = nn.Conv2d(1, 32, kernel_size=3, stride=2, padding=1, bias=False)
        model = nn.Sequential(model.features, nn.AdaptiveAvgPool2d((1, 1)), nn.Flatten())

    elif backbone_name == 'mobilenet_v3_small':
        model = models.mobilenet_v3_small(weights=models.MobileNet_V3_Small_Weights.IMAGENET1K_V1)
        if image_format == "L":
            model.features[0][0] = nn.Conv2d(1, 16, kernel_size=3, stride=2, padding=1, bias=False)
        model = nn.Sequential(model.features, nn.AdaptiveAvgPool2d((1, 1)), nn.Flatten())

    elif backbone_name == 'mobilenet_v3_large':
        model = models.mobilenet_v3_large(weights=models.MobileNet_V3_Large_Weights.IMAGENET1K_V1)
        if image_format == "L":
            model.features[0][0] = nn.Conv2d(1, 16, kernel_size=3, stride=2, padding=1, bias=False)
        model = nn.Sequential(model.features, nn.AdaptiveAvgPool2d((1, 1)), nn.Flatten())

    elif backbone_name == 'efficientnet_b0':
        model = models.efficientnet_b0(weights=models.EfficientNet_B0_Weights.IMAGENET1K_V1)
        if image_format == "L":
            model.features[0][0] = nn.Conv2d(1, 32, kernel_size=3, stride=2, padding=1, bias=False)
        model = nn.Sequential(model.features, nn.AdaptiveAvgPool2d((1, 1)), nn.Flatten())

    elif backbone_name == 'efficientnet_b1':
        model = models.efficientnet_b1(weights=models.EfficientNet_B1_Weights.IMAGENET1K_V1)
        if image_format == "L":
            model.features[0][0] = nn.Conv2d(1, 32, kernel_size=3, stride=2, padding=1, bias=False)
        model = nn.Sequential(model.features, nn.AdaptiveAvgPool2d((1, 1)), nn.Flatten())

    elif backbone_name == 'densenet121':
        model = models.densenet121(weights=models.DenseNet121_Weights.IMAGENET1K_V1)
        if image_format == "L":
            model.features.conv0 = nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False)
        model = nn.Sequential(model.features, nn.AdaptiveAvgPool2d((1, 1)), nn.Flatten())

    elif backbone_name == 'densenet169':
        model = models.densenet169(weights=models.DenseNet169_Weights.IMAGENET1K_V1)
        if image_format == "L":
            model.features.conv0 = nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False)
        model = nn.Sequential(model.features, nn.AdaptiveAvgPool2d((1, 1)), nn.Flatten())

    else:
        raise ValueError(f"Backbone '{backbone_name}' not supported!")

    return model.eval()
