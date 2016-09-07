import os
import numpy as np
from astropy import units as u, constants as const
# Some helper functions


def intensity_black_body_wavelength(wavelength, T):
    wavelength = u.Quantity(wavelength, u.angstrom)
    T = u.Quantity(T, u.K)
    f = ((8 * np.pi * const.h * const.c) / wavelength**5)
    return f / (np.exp((const.h * const.c)/(wavelength * const.k_B * T)) - 1)


def bin_center_to_edge(bin_center):
    hdiff = 0.5 * np.diff(bin_center)
    hdiff = np.hstack((-hdiff[0], hdiff, hdiff[-1]))
    return np.hstack((
        bin_center[0],
        bin_center)) + hdiff


def bin_edge_to_center(bin_edge):
    return 0.5 * (bin_edge[:-1] + bin_edge[1:])


def flux_to_luminosity(flux, distance):
    try:
        return 4 * np.pi * distance.to('cm')**2 * flux
    except AttributeError:
        return 4 * np.pi * u.Quantity(distance, 'Mpc').to('cm')**2 * flux


def set_engines_cpu_affinity():
    import sys
    if sys.platform.startswith('linux'):
        try:
            import psutil
        except ImportError:
            print 'psutil not available - can not set CPU affinity'
        else:
            from multiprocessing import cpu_count
            p = psutil.Process(os.getpid())
            p.cpu_affinity(range(cpu_count()))


def weighted_avg_and_std(values, weights):
    """
    Stolen from
    http://stackoverflow.com/questions/2413522/weighted-standard-deviation-in-numpy
    Return the weighted average and standard deviation.

    values, weights -- Numpy ndarrays with the same shape.
    """
    average = np.average(values, weights=weights)
    variance = np.average((values-average)**2, weights=weights)
    return (average, np.sqrt(variance))

#stolen from https://bitbucket.org/astro_ufsc/pystarlight/src/20c5b3444bbed9ace92e32162bfd816b2e75da3b/src/pystarlight/util/StarlightUtils.py?at=default#cl-93
def ReSamplingMatrixNonUniform(lorig, lresam, extrap = False):
    '''
    Compute resampling matrix R_o2r, useful to convert a spectrum sampled at
    wavelengths lorig to a new grid lresamp. Here, there is no necessity to have constant gris as on :py:func:`ReSamplingMatrix`.
    Input arrays lorig and lresamp are the bin centres of the original and final lambda-grids.
    ResampMat is a Nlresamp x Nlorig matrix, which applied to a vector F_o (with Nlorig entries) returns
    a Nlresamp elements long vector F_r (the resampled spectrum):

        [[ResampMat]] [F_o] = [F_r]

    Warning! lorig and lresam MUST be on ascending order!


    Parameters
    ----------
    lorig : array_like
            Original spectrum lambda array.

    lresam : array_like
             Spectrum lambda array in which the spectrum should be sampled.

    extrap : boolean, optional
           Extrapolate values, i.e., values for lresam < lorig[0]  are set to match lorig[0] and
                                     values for lresam > lorig[-1] are set to match lorig[-1].


    Returns
    -------
    ResampMat : array_like
                Resample matrix.

    Examples
    --------
    >>> lorig = np.linspace(3400, 8900, 9000) * 1.001
    >>> lresam = np.linspace(3400, 8900, 5000)
    >>> forig = np.random.normal(size=len(lorig))**2
    >>> matrix = slut.ReSamplingMatrixNonUniform(lorig, lresam)
    >>> fresam = np.dot(matrix, forig)
    >>> print np.trapz(forig, lorig), np.trapz(fresam, lresam)
    '''

    # Init ResampMatrix
    matrix = np.zeros((len(lresam), len(lorig)))

    # Define lambda ranges (low, upp) for original and resampled.
    lo_low = np.zeros(len(lorig))
    lo_low[1:] = (lorig[1:] + lorig[:-1])/2
    lo_low[0] = lorig[0] - (lorig[1] - lorig[0])/2

    lo_upp = np.zeros(len(lorig))
    lo_upp[:-1] = lo_low[1:]
    lo_upp[-1] = lorig[-1] + (lorig[-1] - lorig[-2])/2

    lr_low = np.zeros(len(lresam))
    lr_low[1:] = (lresam[1:] + lresam[:-1])/2
    lr_low[0] = lresam[0] - (lresam[1] - lresam[0])/2

    lr_upp = np.zeros(len(lresam))
    lr_upp[:-1] = lr_low[1:]
    lr_upp[-1] = lresam[-1] + (lresam[-1] - lresam[-2])/2


    # Iterate over resampled lresam vector
    for i_r in range(len(lresam)):

        # Find in which bins lresam bin within lorig bin
        bins_resam = np.where( (lr_low[i_r] < lo_upp) & (lr_upp[i_r] > lo_low) )[0]

        # On these bins, eval fraction of resamled bin is within original bin.
        for i_o in bins_resam:

            aux = 0

            d_lr = lr_upp[i_r] - lr_low[i_r]
            d_lo = lo_upp[i_o] - lo_low[i_o]
            d_ir = lo_upp[i_o] - lr_low[i_r]  # common section on the right
            d_il = lr_upp[i_r] - lo_low[i_o]  # common section on the left

            # Case 1: resampling window is smaller than or equal to the original window.
            # This is where the bug was: if an original bin is all inside the resampled bin, then
            # all flux should go into it, not then d_lr/d_lo fraction. --Natalia@IoA - 21/12/2012
            if (lr_low[i_r] > lo_low[i_o]) & (lr_upp[i_r] < lo_upp[i_o]):
                aux += 1.

            # Case 2: resampling window is larger than the original window.
            if (lr_low[i_r] < lo_low[i_o]) & (lr_upp[i_r] > lo_upp[i_o]):
                aux += d_lo / d_lr

            # Case 3: resampling window is on the right of the original window.
            if (lr_low[i_r] > lo_low[i_o]) & (lr_upp[i_r] > lo_upp[i_o]):
                aux += d_ir / d_lr

            # Case 4: resampling window is on the left of the original window.
            if (lr_low[i_r] < lo_low[i_o]) & (lr_upp[i_r] < lo_upp[i_o]):
                aux += d_il / d_lr

            matrix[i_r, i_o] += aux


    # Fix matrix to be exactly = 1 ==> TO THINK
    #print np.sum(matrix), np.sum(lo_upp - lo_low), (lr_upp - lr_low).shape


    # Fix extremes: extrapolate if needed
    if (extrap):

        bins_extrapl = np.where( (lr_low < lo_low[0])  )[0]
        bins_extrapr = np.where( (lr_upp > lo_upp[-1]) )[0]

        if (len(bins_extrapl) > 0) & (len(bins_extrapr) > 0):
            io_extrapl = np.where( (lo_low >= lr_low[bins_extrapl[0]])  )[0][0]
            io_extrapr = np.where( (lo_upp <= lr_upp[bins_extrapr[0]])  )[0][-1]

            matrix[bins_extrapl, io_extrapl] = 1.
            matrix[bins_extrapr, io_extrapr] = 1.


    return matrix
