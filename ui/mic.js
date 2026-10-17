// Runs in a 16 kHz AudioContext: applies the gain and posts 80 ms frames of PCM16 with their level.
class Mic extends AudioWorkletProcessor {
  constructor() {
    super();
    this.gain = 1;
    this.reset();
    this.port.onmessage = (e) => { this.gain = e.data.gain; };
  }

  reset() {
    this.buf = new Int16Array(1280);
    this.n = 0;
    this.sum = 0;
    this.peak = 0;
  }

  process(inputs) {
    const x = inputs[0][0];
    if (!x) return true;
    for (let i = 0; i < x.length; i++) {
      const v = Math.max(-1, Math.min(1, x[i] * this.gain));
      this.sum += v * v;
      this.peak = Math.max(this.peak, Math.abs(v));
      this.buf[this.n++] = v * 32767;
      if (this.n === this.buf.length) {
        const rms = Math.sqrt(this.sum / this.n);
        this.port.postMessage({ pcm: this.buf.buffer, rms, peak: this.peak }, [this.buf.buffer]);
        this.reset();
      }
    }
    return true;
  }
}

registerProcessor("mic", Mic);
