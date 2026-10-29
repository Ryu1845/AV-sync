# S3FD face detector, ported from https://github.com/joonson/syncnet_python (MIT)
# Weights: https://www.robots.ox.ac.uk/~vgg/software/lipsync/data/sfd_face.pth

import torch
import torch.nn as nn
import torch.nn.functional as F

IMG_MEAN = (123., 117., 104.)  # RGB
VARIANCE = (0.1, 0.2)
MIN_SIZES = (16, 32, 64, 128, 256, 512)
STEPS = (4, 8, 16, 32, 64, 128)


class L2Norm(nn.Module):

    def __init__(self, n_channels, scale):
        super().__init__()
        self.eps = 1e-10
        self.weight = nn.Parameter(torch.full((n_channels,), float(scale)))

    def forward(self, x):
        norm = x.pow(2).sum(dim=1, keepdim=True).sqrt() + self.eps
        return self.weight[None, :, None, None] * (x / norm)


class S3FDNet(nn.Module):

    def __init__(self):
        super().__init__()

        def conv(c_in, c_out, *args, **kwargs):
            return [nn.Conv2d(c_in, c_out, *args, **kwargs), nn.ReLU(inplace=True)]

        self.vgg = nn.ModuleList([
            *conv(3, 64, 3, 1, padding=1), *conv(64, 64, 3, 1, padding=1),
            nn.MaxPool2d(2, 2),
            *conv(64, 128, 3, 1, padding=1), *conv(128, 128, 3, 1, padding=1),
            nn.MaxPool2d(2, 2),
            *conv(128, 256, 3, 1, padding=1), *conv(256, 256, 3, 1, padding=1), *conv(256, 256, 3, 1, padding=1),
            nn.MaxPool2d(2, 2, ceil_mode=True),
            *conv(256, 512, 3, 1, padding=1), *conv(512, 512, 3, 1, padding=1), *conv(512, 512, 3, 1, padding=1),
            nn.MaxPool2d(2, 2),
            *conv(512, 512, 3, 1, padding=1), *conv(512, 512, 3, 1, padding=1), *conv(512, 512, 3, 1, padding=1),
            nn.MaxPool2d(2, 2),
            *conv(512, 1024, 3, 1, padding=6, dilation=6),
            *conv(1024, 1024, 1, 1),
        ])

        self.L2Norm3_3 = L2Norm(256, 10)
        self.L2Norm4_3 = L2Norm(512, 8)
        self.L2Norm5_3 = L2Norm(512, 5)

        self.extras = nn.ModuleList([
            nn.Conv2d(1024, 256, 1, 1),
            nn.Conv2d(256, 512, 3, 2, padding=1),
            nn.Conv2d(512, 128, 1, 1),
            nn.Conv2d(128, 256, 3, 2, padding=1),
        ])

        channels = (256, 512, 512, 1024, 512, 256)
        self.loc = nn.ModuleList([nn.Conv2d(c, 4, 3, 1, padding=1) for c in channels])
        # the first conf head predicts 3 background scores (max-out) and 1 face score
        self.conf = nn.ModuleList([nn.Conv2d(c, 4 if k == 0 else 2, 3, 1, padding=1) for k, c in enumerate(channels)])

    def forward(self, x):
        """Returns per-prior box offsets (N x P x 4), class probabilities (N x P x 2) and priors (P x 4)."""
        size = x.shape[2:]
        sources = []

        for k, v in enumerate(self.vgg):
            x = v(x)
            if k == 15:
                sources.append(self.L2Norm3_3(x))
            elif k == 22:
                sources.append(self.L2Norm4_3(x))
            elif k == 29:
                sources.append(self.L2Norm5_3(x))
        sources.append(x)

        for k, v in enumerate(self.extras):
            x = F.relu(v(x), inplace=True)
            if k % 2 == 1:
                sources.append(x)

        loc, conf = [], []
        for k, x in enumerate(sources):
            c = self.conf[k](x)
            if k == 0:
                c = torch.cat((c[:, 0:3].amax(dim=1, keepdim=True), c[:, 3:]), dim=1)
            loc.append(self.loc[k](x).permute(0, 2, 3, 1).flatten(1))
            conf.append(c.permute(0, 2, 3, 1).flatten(1))

        loc = torch.cat(loc, 1).view(x.shape[0], -1, 4)
        conf = torch.cat(conf, 1).view(x.shape[0], -1, 2).softmax(-1)
        return loc, conf, prior_boxes(size, [s.shape[2:] for s in sources]).to(loc)


def prior_boxes(image_size, feature_maps):
    """Prior boxes as (cx, cy, w, h), relative to the image size."""
    imh, imw = image_size
    priors = []
    for (feath, featw), min_size, step in zip(feature_maps, MIN_SIZES, STEPS):
        cy = (torch.arange(feath, dtype=torch.float64) + 0.5) / (imh / step)
        cx = (torch.arange(featw, dtype=torch.float64) + 0.5) / (imw / step)
        cy, cx = torch.meshgrid(cy, cx, indexing='ij')
        priors.append(torch.stack([cx, cy, torch.full_like(cx, min_size / imw), torch.full_like(cy, min_size / imh)], -1))
    return torch.cat([p.view(-1, 4) for p in priors]).float()


def decode(loc, priors):
    """Undo the box offset encoding; returns (x1, y1, x2, y2) relative to the image size."""
    boxes = torch.cat((
        priors[:, :2] + loc[:, :2] * VARIANCE[0] * priors[:, 2:],
        priors[:, 2:] * torch.exp(loc[:, 2:] * VARIANCE[1])), 1)
    boxes[:, :2] -= boxes[:, 2:] / 2
    boxes[:, 2:] += boxes[:, :2]
    return boxes


def nms(boxes, scores, thresh):
    """Greedy non-maximum suppression, returns the indices of the kept boxes."""
    x1, y1, x2, y2 = boxes.unbind(1)
    areas = (x2 - x1) * (y2 - y1)
    order = scores.argsort(descending=True)
    keep = []
    while order.numel() > 0:
        i, order = order[0], order[1:]
        keep.append(i.item())
        w = (torch.minimum(x2[i], x2[order]) - torch.maximum(x1[i], x1[order])).clamp(min=0)
        h = (torch.minimum(y2[i], y2[order]) - torch.maximum(y1[i], y1[order])).clamp(min=0)
        inter = w * h
        order = order[inter / (areas[i] + areas[order] - inter) <= thresh]
    return torch.tensor(keep, dtype=torch.long)


class S3FD:

    def __init__(self, path, device):
        self.device = device
        self.net = S3FDNet().to(device).eval()
        self.net.load_state_dict(torch.load(path, map_location=device, weights_only=True))

    def detect(self, image, conf_th=0.9, scale=0.25):
        """image: H x W x 3 uint8 RGB. Returns N x 5 rows of (y1, x1, y2, x2, score), coordinates relative to the image."""
        return self.detect_batch([image], conf_th, scale)[0]

    @torch.inference_mode()
    def detect_batch(self, images, conf_th=0.9, scale=0.25):
        """images: equally sized H x W x 3 uint8 RGB frames, run through the network together. Returns a list of detect()."""
        # move uint8 to the device and convert there: 4x less host to device traffic
        x = torch.stack(images).to(self.device).permute(0, 3, 1, 2).float()
        if scale != 1:
            x = F.interpolate(x, scale_factor=scale, mode='bilinear')
        # contiguous NCHW for the network, as with a single permuted frame (channels last selects other conv kernels)
        x = x.contiguous() - torch.tensor(IMG_MEAN, device=self.device).view(1, 3, 1, 1)
        loc, conf, priors = self.net(x)

        dets = []
        for loc_i, conf_i in zip(loc, conf):
            scores = conf_i[:, 1]
            mask = scores > conf_th
            boxes, scores = decode(loc_i[mask], priors[mask]), scores[mask]

            keep = nms(boxes, scores, 0.3)
            boxes, scores = boxes[keep], scores[keep]
            keep = nms(boxes, scores, 0.1)
            boxes, scores = boxes[keep], scores[keep]

            dets.append(torch.cat([boxes[:, [1, 0, 3, 2]], scores[:, None]], 1).cpu())
        return dets
