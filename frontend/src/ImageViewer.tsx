import { useEffect, useRef, useState } from 'react';
import { Focus, Minus, Plus, RotateCcw } from 'lucide-react';
import type { ArtifactSet, SulfideMaskType } from './types';

export type OverlayState = {
  talc: boolean;
  sulfide: boolean;
  coarse: boolean;
  sulfideMaskType: SulfideMaskType;
  talcOpacity: number;
  sulfideOpacity: number;
  coarseOpacity: number;
};

export function ImageViewer({
  artifacts,
  overlays,
  filename,
}: {
  artifacts: ArtifactSet;
  overlays: OverlayState;
  filename: string;
}) {
  const [scale, setScale] = useState(1);
  const [position, setPosition] = useState({ x: 0, y: 0 });
  const drag = useRef<{ x: number; y: number; px: number; py: number } | null>(null);
  const viewer = useRef<HTMLDivElement>(null);

  useEffect(() => {
    setScale(1);
    setPosition({ x: 0, y: 0 });
  }, [filename]);

  useEffect(() => {
    const element = viewer.current;
    if (!element) return;
    const onWheel = (event: WheelEvent) => {
      event.preventDefault();
      setScale((current) => Math.min(5, Math.max(0.5, current + (event.deltaY > 0 ? -0.15 : 0.15))));
    };
    element.addEventListener('wheel', onWheel, { passive: false });
    return () => element.removeEventListener('wheel', onWheel);
  }, []);

  const zoom = (delta: number) => setScale((current) => Math.min(5, Math.max(0.5, current + delta)));
  const reset = () => {
    setScale(1);
    setPosition({ x: 0, y: 0 });
  };
  const original = artifacts.original;
  const sulfideOverlay =
    overlays.sulfideMaskType === 'sam' && artifacts.sulfide_sam_overlay
      ? artifacts.sulfide_sam_overlay
      : artifacts.sulfide_cv_overlay || artifacts.sulfide_mask;

  return (
    <section className="viewer-shell">
      <div className="viewer-toolbar" aria-label="Масштаб изображения">
        <button type="button" onClick={() => zoom(-0.25)} aria-label="Уменьшить">
          <Minus size={17} />
        </button>
        <span>{Math.round(scale * 100)}%</span>
        <button type="button" onClick={() => zoom(0.25)} aria-label="Увеличить">
          <Plus size={17} />
        </button>
        <button type="button" onClick={reset} aria-label="Сбросить масштаб">
          <RotateCcw size={16} />
        </button>
      </div>
      <div
        ref={viewer}
        className="viewer"
        onPointerDown={(event) => {
          event.currentTarget.setPointerCapture(event.pointerId);
          drag.current = { x: event.clientX, y: event.clientY, px: position.x, py: position.y };
        }}
        onPointerMove={(event) => {
          if (!drag.current) return;
          setPosition({
            x: drag.current.px + event.clientX - drag.current.x,
            y: drag.current.py + event.clientY - drag.current.y,
          });
        }}
        onPointerUp={() => {
          drag.current = null;
        }}
        onPointerCancel={() => {
          drag.current = null;
        }}
      >
        {original ? (
          <div
            className="image-stack"
            style={{ transform: `translate(${position.x}px, ${position.y}px) scale(${scale})` }}
          >
            <img src={original} alt={`Микроскопический снимок ${filename}`} draggable={false} />
            {overlays.coarse && (artifacts.coarse_overlay || artifacts.coarse_mask || artifacts.segmentation_mask) && (
              <img
                className="overlay"
                src={artifacts.coarse_overlay || artifacts.coarse_mask || artifacts.segmentation_mask}
                alt=""
                draggable={false}
                style={{ opacity: overlays.coarseOpacity }}
              />
            )}
            {overlays.sulfide && sulfideOverlay && (
              <img
                className="overlay"
                src={sulfideOverlay}
                alt=""
                draggable={false}
                style={{ opacity: overlays.sulfideOpacity }}
              />
            )}
            {overlays.talc && (artifacts.talc_overlay || artifacts.talc_mask || artifacts.refined_talc_mask) && (
              <img
                className="overlay"
                src={artifacts.talc_overlay || artifacts.talc_mask || artifacts.refined_talc_mask}
                alt=""
                draggable={false}
                style={{ opacity: overlays.talcOpacity }}
              />
            )}
          </div>
        ) : (
          <div className="viewer-empty">
            <Focus size={28} />
            <p>Исходное изображение недоступно</p>
          </div>
        )}
      </div>
      <p className="viewer-hint">Колесо — масштаб · перетаскивание — панорама</p>
    </section>
  );
}
