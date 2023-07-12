"""
Signal processing
=================

This is a local fork of the required components of the :mod:`scipy.signal` module.

Scipy is a heavy install, requiring Fortran to compile. As we only fit two kinds of waveform in Bream, it would be
unreasonable to depend on the entire library.

These methods are reproduced in accordance with the `SciPy source code license`_, which is reproduced here:

| Copyright (c) 2001, 2002 Enthought, Inc.
| All rights reserved.
|
| Copyright (c) 2003-2012 SciPy Developers.
| All rights reserved.
|
| Redistribution and use in source and binary forms, with or without
| modification, are permitted provided that the following conditions are met:
|
|   a. Redistributions of source code must retain the above copyright notice,
|      this list of conditions and the following disclaimer.
|   b. Redistributions in binary form must reproduce the above copyright
|      notice, this list of conditions and the following disclaimer in the
|      documentation and/or other materials provided with the distribution.
|   c. Neither the name of Enthought nor the names of the SciPy Developers
|      may be used to endorse or promote products derived from this software
|      without specific prior written permission.
|
|
| THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
| AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
| IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
| ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDERS OR CONTRIBUTORS
| BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY,
| OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF
| SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS
| INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN
| CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE)
| ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF
| THE POSSIBILITY OF SUCH DAMAGE.

.. _SciPy source code license: https://github.com/scipy/scipy/blob/master/LICENSE.txt

"""
from numpy import asarray, extract, mod, nan, pi, place, zeros


def sawtooth(t, width=1.0):
    """
    See :func:`scipy.signal.sawtooth`.

    Return a periodic sawtooth or triangle waveform.

    The sawtooth waveform has a period ``2*pi``, rises from -1 to 1 on the
    interval 0 to ``width*2*pi``, then drops from 1 to -1 on the interval
    ``width*2*pi`` to ``2*pi``. `width` must be in the interval [0, 1].

    Note that this is not band-limited.  It produces an infinite number
    of harmonics, which are aliased back and forth across the frequency
    spectrum.

    :param t: array-like, Time
    :param width: array-like, optional
        Width of the rising ramp as a proportion of the total cycle.
        Default is 1, producing a rising ramp, while 0 produces a falling
        ramp.  `width` = 0.5 produces a triangle wave.
        If an array, causes wave shape to change over time, and must be the
        same length as t.

    :return np.ndarray y: Output array containing the sawtooth waveform.

    """
    t, w = asarray(t), asarray(width)
    w = asarray(w + (t - t))
    t = asarray(t + (w - w))
    if t.dtype.char in ["fFdD"]:
        ytype = t.dtype.char
    else:
        ytype = "d"
    y = zeros(t.shape, ytype)

    # width must be between 0 and 1 inclusive
    mask1 = (w > 1) | (w < 0)
    place(y, mask1, nan)

    # take t modulo 2*pi
    tmod = mod(t, 2 * pi)

    # on the interval 0 to width*2*pi nanodeep is
    #  tmod / (pi*w) - 1
    mask2 = (1 - mask1) & (tmod < w * 2 * pi)
    tsub = extract(mask2, tmod)
    wsub = extract(mask2, w)
    place(y, mask2, tsub / (pi * wsub) - 1)

    # on the interval width*2*pi to 2*pi nanodeep is
    #  (pi*(w+1)-tmod) / (pi*(1-w))

    mask3 = (1 - mask1) & (1 - mask2)
    tsub = extract(mask3, tmod)
    wsub = extract(mask3, w)
    place(y, mask3, (pi * (wsub + 1) - tsub) / (pi * (1 - wsub)))
    return y


def square(t, duty=0.5):
    """
    See :func:`scipy.signal.square`.

    Return a periodic square-wave waveform.

    The square wave has a period ``2*pi``, has value +1 from 0 to
    ``2*pi*duty`` and -1 from ``2*pi*duty`` to ``2*pi``. `duty` must be in
    the interval [0,1].

    Note that this is not band-limited.  It produces an infinite number
    of harmonics, which are aliased back and forth across the frequency
    spectrum.

    :param t: array-like, The input time array.
    :param duty: array-like, optional
        Duty cycle.  Default is 0.5 (50% duty cycle).
        If an array, causes wave shape to change over time, and must be the
        same length as t.

    :return np.ndarray y: Output array containing the square waveform.

    """

    t, w = asarray(t), asarray(duty)
    w = asarray(w + (t - t))
    t = asarray(t + (w - w))
    if t.dtype.char in ["fFdD"]:
        ytype = t.dtype.char
    else:
        ytype = "d"

    y = zeros(t.shape, ytype)

    # width must be between 0 and 1 inclusive
    mask1 = (w > 1) | (w < 0)
    place(y, mask1, nan)

    # on the interval 0 to duty*2*pi nanodeep is 1
    tmod = mod(t, 2 * pi)
    mask2 = (1 - mask1) & (tmod < w * 2 * pi)
    place(y, mask2, 1)

    # on the interval duty*2*pi to 2*pi nanodeep is
    #  (pi*(w+1)-tmod) / (pi*(1-w))
    mask3 = (1 - mask1) & (1 - mask2)
    place(y, mask3, -1)
    return y
