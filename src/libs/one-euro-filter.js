// Ref: https://jaantollander.com/post/noise-filtering-using-one-euro-filter/#mjx-eqn%3A1

const smoothingFactor = (te, cutoff) => {
  const r = 2 * Math.PI * cutoff * te;
  return r / (r+1);
}

const exponentialSmoothing = (a, x, xPrev) => {
  return a * x + (1 - a) * xPrev;
}

class OneEuroFilter {
  constructor({minCutOff, beta}) {
    this.minCutOff = minCutOff;
    this.beta = beta;
    this.dCutOff = 0.001; // period in milliseconds, so default to 0.001 = 1Hz

    this.xPrev = null;
    this.dxPrev = null;
    this.tPrev = null;
    this.initialized = false;

    // Pre-allocated arrays for reuse (will be sized on first filter call)
    this.dx = null;
    this.dxHat = null;
    this.xHat = null;
  }

  reset() {
    this.initialized = false;
    this.xPrev = null;
    this.dxPrev = null;
    this.tPrev = null;
  }

  filter(t, x) {
    if (!this.initialized) {
      this.initialized = true;
      this.xPrev = [...x];
      this.dxPrev = new Array(x.length).fill(0);
      this.tPrev = t;

      // Pre-allocate working arrays
      this.dx = new Array(x.length);
      this.dxHat = new Array(x.length);
      this.xHat = new Array(x.length);

      return x;
    }

    const {xPrev, tPrev, dxPrev, dx, dxHat, xHat} = this;

    const te = t - tPrev;

    // Handle edge case: same timestamp
    if (te <= 0) {
      return xPrev;
    }

    const ad = smoothingFactor(te, this.dCutOff);
    const teInv = 1 / te; // Pre-compute for multiplication

    for (let i = 0; i < x.length; i++) {
      // The filtered derivative of the signal.
      dx[i] = (x[i] - xPrev[i]) * teInv;
      dxHat[i] = exponentialSmoothing(ad, dx[i], dxPrev[i]);

      // The filtered signal
      const cutOff = this.minCutOff + this.beta * Math.abs(dxHat[i]);
      const a = smoothingFactor(te, cutOff);
      xHat[i] = exponentialSmoothing(a, x[i], xPrev[i]);
    }

    // update prev - swap references instead of copying
    const temp = this.xPrev;
    this.xPrev = this.xHat;
    this.xHat = temp;

    const temp2 = this.dxPrev;
    this.dxPrev = this.dxHat;
    this.dxHat = temp2;

    this.tPrev = t;

    return this.xPrev;
  }
}

export {
  OneEuroFilter
}
