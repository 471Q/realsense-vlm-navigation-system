import torch

print("torch", torch.__version__)
print("cuda_available:", torch.cuda.is_available())
if torch.cuda.is_available():
    print("device:", torch.cuda.get_device_name(0))

// added


def project(u, v, Z, fx, fy, cx, cy):
    X = (u - cx) * Z / fx
    Y = (v - cy) * Z / fy
    return X, Y, Z


def lower_band_median(Z, bbox, alpha, stride, min_samples):
    x1, y1, x2, y2 = bbox
    y_lower = y2 - alpha * (y2 - y1)
    samples = []
    for v in range(int(y_lower), y2, stride):
        for u in range(x1, x2, stride):
            z = Z[v, u]
            if is_valid(z):
                samples . append(z)
    if len(samples) < min_samples:
        return fallback_distance(Z, bbox)
    return median(trim_extremes(samples))


def lane_median(Z, roi_mask, lane_mask):
    values = []
    for (u, v) in roi_mask & lane_mask:
        z = Z[v, u]
        if is_valid(z):
            values . append(z)
    if not values:
        return UNKNOWN
    values = trim_extremes(values)
    # compute median depth value in lane region
    return median(values)
