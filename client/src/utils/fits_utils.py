
from astropy.io import fits
from astropy.wcs import WCS

def get_image_dim(image_path):
    # Open the image and get the header
    with fits.open(image_path) as hdul:
        header = hdul[0].header
        data = hdul[0].data

        if data is None:
            info = {'type': 'No data', 'shape': (), 'axes': []}

        shape = tuple(s for s in data.shape if s > 1) # Remove "1" dimensions
        ndim = len(shape)

        if ndim == 2:
            image_type = '2D image'
        elif ndim == 3:
            image_type = '3D image'
        else:
            image_type = f'Unsupported ({ndim}D)'

        try:
            wcs = WCS(header)
            ctype = [wcs.wcs.ctype[i] for i in range(wcs.naxis) if data.shape[::-1][i] > 1]
        except Exception:
            ctype = []

        info = {
            'type': image_type,
            'shape': shape,
            'axes': ctype
        }