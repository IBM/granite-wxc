import time
import numpy as np
import scipy.stats as stats

import matplotlib.pyplot as plt
import matplotlib.colors as colors
import matplotlib.animation as animation

def plot_spatial(sample, ax, title, **kwargs):
    
    vmin=kwargs.get('vmin', np.min(sample))
    vmax=kwargs.get('vmax', np.max(sample))
    
    if kwargs.get('c_lognorm', False):
        vmin = max(vmin, 0)
        e = kwargs.get('e', 1e-44)
        vmin+=e
        norm=colors.LogNorm(vmin=vmin, vmax=vmax)
    else:
        norm=colors.Normalize(vmin=vmin, vmax=vmax)
        
    im = ax.imshow(
        sample,
        cmap=kwargs.get('cmap', 'coolwarm'),
        norm=norm,
        origin='lower',
        animated=kwargs.get('animated', False),
    )
    ax.set_xlabel('Longitudes')
    ax.set_ylabel('Latitudes')
    ax.tick_params(axis='both', which='major', labelsize=10)
    ax.title.set_text(title)
    
    return im

def plot_model_residual(sample, sample_id, title, **kwargs):

    fig, ax = plt.subplots(figsize=(15, 5))
    plt.suptitle(title)
    
    im = plot_spatial(sample, ax, sample_id,  **kwargs)
    
    label = f"{kwargs.get('var_name_tile', '')} [{kwargs.get('var_unit', '')}]"
    fig.colorbar(im, ax=ax, orientation='vertical', label=label, fraction=0.05, aspect=50)
    
    

def plot_model_results(samples, samples_id, title, **kwargs):
    
    fig, axes = plt.subplots(1, len(samples), figsize=(20, 5))
    plt.suptitle(title)
    
    for i, ax in enumerate(axes):
        if samples_id[i] == 'Residual':
            im_res = plot_spatial(samples[i], ax, samples_id[i],  **kwargs.get('plot_residual_kwargs'))
            label = f"{kwargs.get('var_name_tile', '')} [{kwargs.get('var_unit', '')}]"
            fig.colorbar(im_res, ax=ax, orientation='vertical', label=label, fraction=0.015)
        else:
            im = plot_spatial(samples[i], ax, samples_id[i],  **kwargs)
        
    label = f"{kwargs.get('var_name_tile', '')} [{kwargs.get('var_unit', '')}]"
    fig.colorbar(im, ax=axes, orientation='horizontal', label=label, fraction=0.05, aspect=50)
 
    plt.close()
    
    return fig

def plot_power_spectrum(img, ax, save_fig=False):
    """
    A power spctrum mesaures the strength of features at different resolutions

    :param img: H x W
    """

    npix = img.shape[-2], img.shape[-1]

    fft_img = np.fft.fftn(img)
    fft_amp = np.abs(fft_img)**2
    fft_amp = fft_amp.flatten()

    kfreq_x = np.fft.fftfreq(npix[1]) * npix[1] # wave vector
    kfreq_y = np.fft.fftfreq(npix[0]) * npix[0] # wave vector
    kfreq2D = np.meshgrid(kfreq_x, kfreq_y)
    knrm = np.sqrt(kfreq2D[0]**2 + kfreq2D[1]**2)
    knrm = knrm.flatten()

    kbins = np.arange(0.5, min(*npix)//2+1, 1.)
    kvals = 0.5 * (kbins[1:] + kbins[:-1])
    Abins, _, _ = stats.binned_statistic(
        knrm,
        fft_amp,
        statistic='mean',
        bins=kbins
    )

    Abins*= np.pi * (kbins[1:]**2 - kbins[:-1]**2)

    ax.loglog(kvals, Abins)
    ax.set_xlabel("$k$")
    ax.set_ylabel("$P(k)$")
    plt.tight_layout()

    if save_fig:
        timestr = time.strftime("%Y%m%d-%H%M")
        plt.savefig(f'power_spectrum_{timestr}.png', dpi=300, bbox_inches='tight')

def spatial_rmse(y_hat, y):
    return np.mean((y_hat - y) ** 2) ** 0.5
    
def spatial_bias(y_hat, y):
    return y_hat.mean() - y.mean()