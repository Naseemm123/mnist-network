export function relu(value) { return value > 0 ? value : 0; }

export function forwardProp(weights, input) {
  const z1 = weights.w1.map((row, i) => row.reduce((sum, value, j) => sum + value * input[j], weights.b1[i]));
  const a1 = z1.map(relu);
  const z2 = weights.w2.map((row, i) => row.reduce((sum, value, j) => sum + value * a1[j], weights.b2[i]));
  const maxLogit = Math.max(...z2);
  const exponentials = z2.map(value => Math.exp(value - maxLogit));
  const total = exponentials.reduce((sum, value) => sum + value, 0);
  return { z1, a1, z2, a2: exponentials.map(value => value / total) };
}

export async function loadWeights() {
  const response = await fetch(`${import.meta.env.BASE_URL}weights.json`);
  if (!response.ok) throw new Error('weights.json could not be loaded');
  return response.json();
}

export function normalizeDrawing(pixels) {
  const points = [];
  let total = 0, sumX = 0, sumY = 0;
  for (let y = 0; y < 28; y++) for (let x = 0; x < 28; x++) {
    const ink = pixels[y * 28 + x];
    if (ink > 0.02) points.push([x, y, ink]);
    total += ink; sumX += x * ink; sumY += y * ink;
  }
  if (!points.length || total === 0) return new Array(784).fill(0);
  const minX = Math.max(0, Math.min(...points.map(point => point[0])) - 1);
  const maxX = Math.min(27, Math.max(...points.map(point => point[0])) + 1);
  const minY = Math.max(0, Math.min(...points.map(point => point[1])) - 1);
  const maxY = Math.min(27, Math.max(...points.map(point => point[1])) + 1);
  const size = Math.max(maxX - minX + 1, maxY - minY + 1);
  const scaled = new Array(784).fill(0);
  for (const [x, y, ink] of points) {
    const nx = Math.round(13.5 + ((x - (minX + maxX) / 2) / size) * 20);
    const ny = Math.round(13.5 + ((y - (minY + maxY) / 2) / size) * 20);
    if (nx >= 0 && nx < 28 && ny >= 0 && ny < 28) scaled[ny * 28 + nx] = Math.max(scaled[ny * 28 + nx], ink);
  }
  const next = new Array(784).fill(0);
  let mass = 0, cx = 0, cy = 0;
  scaled.forEach((ink, i) => { const x = i % 28, y = Math.floor(i / 28); mass += ink; cx += x * ink; cy += y * ink; });
  const shiftX = 13.5 - cx / mass, shiftY = 13.5 - cy / mass;
  scaled.forEach((ink, i) => { if (!ink) return; const x = Math.round(i % 28 + shiftX), y = Math.round(Math.floor(i / 28) + shiftY); if (x >= 0 && x < 28 && y >= 0 && y < 28) next[y * 28 + x] = Math.max(next[y * 28 + x], ink); });
  return next;
}
