import { useEffect, useRef } from 'react';

export default function Canvas({ pixels, onChange }) {
  const canvasRef = useRef(null);
  const drawing = useRef(false);
  useEffect(() => {
    const canvas = canvasRef.current, ctx = canvas.getContext('2d');
    ctx.clearRect(0, 0, 280, 280); ctx.fillStyle = '#fbf8ef'; ctx.fillRect(0, 0, 280, 280);
    pixels.forEach((value, i) => { if (!value) return; ctx.fillStyle = `rgba(18, 25, 30, ${value})`; ctx.fillRect((i % 28) * 10, Math.floor(i / 28) * 10, 10, 10); });
    ctx.strokeStyle = 'rgba(18,25,30,.08)'; ctx.lineWidth = 1;
    for (let i = 0; i <= 280; i += 10) { ctx.beginPath(); ctx.moveTo(i, 0); ctx.lineTo(i, 280); ctx.stroke(); ctx.beginPath(); ctx.moveTo(0, i); ctx.lineTo(280, i); ctx.stroke(); }
  }, [pixels]);
  const paint = event => {
    const rect = canvasRef.current.getBoundingClientRect();
    const x = Math.max(0, Math.min(27, Math.floor(((event.clientX - rect.left) / rect.width) * 28)));
    const y = Math.max(0, Math.min(27, Math.floor(((event.clientY - rect.top) / rect.height) * 28)));
    const next = [...pixels];
    for (let dy = -1; dy <= 1; dy++) for (let dx = -1; dx <= 1; dx++) { const px = x + dx, py = y + dy; if (px >= 0 && px < 28 && py >= 0 && py < 28) next[py * 28 + px] = Math.max(next[py * 28 + px], dx === 0 && dy === 0 ? 1 : .55); }
    onChange(next);
  };
  return <canvas ref={canvasRef} className="drawing-canvas" width="280" height="280" onPointerDown={event => { drawing.current = true; event.currentTarget.setPointerCapture(event.pointerId); paint(event); }} onPointerMove={event => drawing.current && paint(event)} onPointerUp={() => { drawing.current = false; }} onPointerCancel={() => { drawing.current = false; }} aria-label="28 by 28 digit drawing canvas" />;
}
