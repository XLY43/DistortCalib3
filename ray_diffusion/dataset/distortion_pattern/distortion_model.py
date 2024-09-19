import torch

def distortionParameter(types):
    parameters = []

    if types == 'barrel':
        Lambda = torch.rand(1).item() * -5e-5 / 4
        x0 = 448
        y0 = 448
        parameters.extend([Lambda, x0, y0])

    elif types == 'pincushion':
        Lambda = torch.rand(1).item() * 8.6e-5 / 4
        x0 = 448
        y0 = 448
        parameters.extend([Lambda, x0, y0])

    elif types == 'shear':
        shear = torch.rand(1).item() * 0.8 - 0.4
        parameters.append(shear)

    elif types == 'projective':
        x1 = 0
        x4 = torch.rand(1).item() * 0.1 + 0.1

        x2 = 1 - x1
        x3 = 1 - x4

        y1 = 0.005
        y4 = 1 - y1
        y2 = y1
        y3 = y4

        a31 = ((x1 - x2 + x3 - x4) * (y4 - y3) - (y1 - y2 + y3 - y4) * (x4 - x3)) / ((x2 - x3) * (y4 - y3) - (x4 - x3) * (y2 - y3))
        a32 = ((y1 - y2 + y3 - y4) * (x2 - x3) - (x1 - x2 + x3 - x4) * (y2 - y3)) / ((x2 - x3) * (y4 - y3) - (x4 - x3) * (y2 - y3))

        a11 = x2 - x1 + a31 * x2
        a12 = x4 - x1 + a32 * x4
        a13 = x1

        a21 = y2 - y1 + a31 * y2
        a22 = y4 - y1 + a32 * y4
        a23 = y1

        parameters.extend([a11, a12, a13, a21, a22, a23, a31, a32])

    return parameters

def distortionModel(types, xd, yd, W, H, parameter):
    if types in ['barrel', 'pincushion']:
        Lambda = parameter[0]
        x0 = parameter[1]
        y0 = parameter[2]
        coeff = 1 + Lambda * ((xd - x0) ** 2 + (yd - y0) ** 2)
        if coeff == 0:
            xu = W
            yu = H
        else:
            xu = (xd - x0) / coeff + x0
            yu = (yd - y0) / coeff + y0
        return xu, yu

    elif types == 'shear':
        shear = parameter[0]
        xu = xd + shear * yd - shear * W / 2
        yu = yd
        return xu, yu

    elif types == 'projective':
        a11 = parameter[0]
        a12 = parameter[1]
        a13 = parameter[2]
        a21 = parameter[3]
        a22 = parameter[4]
        a23 = parameter[5]
        a31 = parameter[6]
        a32 = parameter[7]
        im = xd / (W - 1.0)
        jm = yd / (H - 1.0)
        xu = (W - 1.0) * (a11 * im + a12 * jm + a13) / (a31 * im + a32 * jm + 1)
        yu = (H - 1.0) * (a21 * im + a22 * jm + a23) / (a31 * im + a32 * jm + 1)
        return xu, yu