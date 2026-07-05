import { useEffect, useRef, useState } from 'react';
import {
  Check,
  Eraser,
  Focus,
  Minus,
  PenTool,
  Plus,
  RotateCcw,
  Save,
  Trash2,
  Undo2,
  X,
} from 'lucide-react';
import type {
  ArtifactSet,
  SulfideMaskType,
  TalcEditPoint,
  TalcEditPolygon,
} from './types';

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
  manualEdits = [],
  manualEditRevision,
  editable = false,
  onSaveEdits,
  onResetEdits,
}: {
  artifacts: ArtifactSet;
  overlays: OverlayState;
  filename: string;
  manualEdits?: TalcEditPolygon[];
  manualEditRevision?: string;
  editable?: boolean;
  onSaveEdits?: (polygons: TalcEditPolygon[]) => Promise<void>;
  onResetEdits?: () => Promise<void>;
}) {
  const [scale, setScale] = useState(1);
  const [position, setPosition] = useState({ x: 0, y: 0 });
  const [imageSize, setImageSize] = useState({ width: 1, height: 1 });
  const [editing, setEditing] = useState(false);
  const [operation, setOperation] = useState<'add' | 'remove'>('add');
  const [polygons, setPolygons] = useState<TalcEditPolygon[]>(manualEdits);
  const [currentPoints, setCurrentPoints] = useState<TalcEditPoint[]>([]);
  const [saving, setSaving] = useState(false);
  const [editError, setEditError] = useState<string | null>(null);
  const drag = useRef<{ x: number; y: number; px: number; py: number } | null>(null);
  const viewer = useRef<HTMLDivElement>(null);

  useEffect(() => {
    setScale(1);
    setPosition({ x: 0, y: 0 });
    setEditing(false);
    setCurrentPoints([]);
  }, [filename]);

  useEffect(() => {
    setPolygons(manualEdits);
  }, [filename, manualEditRevision]);

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
  const revision = manualEditRevision
    ? `manual=${encodeURIComponent(manualEditRevision)}`
    : '';
  const revised = (source: string | undefined) => {
    if (!source || !revision) return source;
    return `${source}${source.includes('?') ? '&' : '?'}${revision}`;
  };
  const closePolygon = () => {
    if (currentPoints.length < 3) return;
    setPolygons((current) => [
      ...current,
      { operation, points: currentPoints },
    ]);
    setCurrentPoints([]);
  };
  const cancelEditing = () => {
    setPolygons(manualEdits);
    setCurrentPoints([]);
    setEditError(null);
    setEditing(false);
  };

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
        {editable && !editing && (
          <button
            type="button"
            onClick={() => {
              setPolygons(manualEdits);
              setCurrentPoints([]);
              setEditError(null);
              setEditing(true);
            }}
            aria-label="Редактировать маску талька"
            title="Редактировать маску талька"
          >
            <PenTool size={16} />
          </button>
        )}
      </div>
      {editing && (
        <div className="mask-editor-toolbar" aria-label="Редактор маски талька">
          <div className="mask-editor-modes">
            <button
              type="button"
              className={operation === 'add' ? 'active add' : ''}
              onClick={() => {
                setOperation('add');
                setCurrentPoints([]);
              }}
            >
              <Plus size={14} /> Добавить
            </button>
            <button
              type="button"
              className={operation === 'remove' ? 'active remove' : ''}
              onClick={() => {
                setOperation('remove');
                setCurrentPoints([]);
              }}
            >
              <Eraser size={14} /> Удалить
            </button>
          </div>
          <span>{currentPoints.length ? `Вершин: ${currentPoints.length}` : `Полигонов: ${polygons.length}`}</span>
          <button
            type="button"
            disabled={currentPoints.length < 3}
            onClick={closePolygon}
          >
            <Check size={14} /> Замкнуть
          </button>
          <button
            type="button"
            disabled={!currentPoints.length && !polygons.length}
            onClick={() => {
              if (currentPoints.length) {
                setCurrentPoints((current) => current.slice(0, -1));
              } else {
                setPolygons((current) => current.slice(0, -1));
              }
            }}
          >
            <Undo2 size={14} /> Отменить
          </button>
          <button
            type="button"
            className="save"
            disabled={saving || currentPoints.length > 0}
            onClick={async () => {
              if (!onSaveEdits) return;
              setSaving(true);
              setEditError(null);
              try {
                await onSaveEdits(polygons);
                setEditing(false);
              } catch (reason) {
                setEditError(reason instanceof Error ? reason.message : 'Не удалось сохранить маску.');
              } finally {
                setSaving(false);
              }
            }}
          >
            <Save size={14} /> {saving ? 'Сохранение…' : 'Сохранить'}
          </button>
          {manualEdits.length > 0 && (
            <button
              type="button"
              disabled={saving}
              onClick={async () => {
                if (!onResetEdits) return;
                setSaving(true);
                setEditError(null);
                try {
                  await onResetEdits();
                  setPolygons([]);
                  setCurrentPoints([]);
                  setEditing(false);
                } catch (reason) {
                  setEditError(reason instanceof Error ? reason.message : 'Не удалось сбросить правки.');
                } finally {
                  setSaving(false);
                }
              }}
            >
              <Trash2 size={14} /> Сбросить правки
            </button>
          )}
          <button type="button" onClick={cancelEditing} disabled={saving}>
            <X size={14} /> Закрыть
          </button>
          {editError && <b role="alert">{editError}</b>}
        </div>
      )}
      <div
        ref={viewer}
        className={`viewer ${editing ? 'editing-mask' : ''}`}
        onPointerDown={(event) => {
          if (editing) return;
          event.currentTarget.setPointerCapture(event.pointerId);
          drag.current = { x: event.clientX, y: event.clientY, px: position.x, py: position.y };
        }}
        onPointerMove={(event) => {
          if (editing) return;
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
            <img
              src={original}
              alt={`Микроскопический снимок ${filename}`}
              draggable={false}
              onLoad={(event) => setImageSize({
                width: event.currentTarget.naturalWidth,
                height: event.currentTarget.naturalHeight,
              })}
            />
            {overlays.coarse && (artifacts.coarse_overlay || artifacts.coarse_mask || artifacts.segmentation_mask) && (
              <img
                className="overlay"
                src={revised(artifacts.coarse_overlay || artifacts.coarse_mask || artifacts.segmentation_mask)}
                alt=""
                draggable={false}
                style={{ opacity: overlays.coarseOpacity }}
              />
            )}
            {overlays.sulfide && sulfideOverlay && (
              <img
                className="overlay"
                src={revised(sulfideOverlay)}
                alt=""
                draggable={false}
                style={{ opacity: overlays.sulfideOpacity }}
              />
            )}
            {overlays.talc && (artifacts.talc_overlay || artifacts.talc_mask || artifacts.refined_talc_mask) && (
              <img
                className="overlay"
                src={revised(artifacts.talc_overlay || artifacts.talc_mask || artifacts.refined_talc_mask)}
                alt=""
                draggable={false}
                style={{ opacity: overlays.talcOpacity }}
              />
            )}
            {editing && (
              <svg
                className="mask-editor-canvas"
                viewBox={`0 0 ${imageSize.width} ${imageSize.height}`}
                preserveAspectRatio="none"
                aria-label="Область рисования полигонов"
                onPointerDown={(event) => {
                  event.preventDefault();
                  event.stopPropagation();
                  const bounds = event.currentTarget.getBoundingClientRect();
                  if (!bounds.width || !bounds.height) return;
                  setCurrentPoints((current) => [
                    ...current,
                    {
                      x: Math.max(0, Math.min(1, (event.clientX - bounds.left) / bounds.width)),
                      y: Math.max(0, Math.min(1, (event.clientY - bounds.top) / bounds.height)),
                    },
                  ]);
                }}
              >
                {polygons.map((polygon, index) => (
                  <polygon
                    key={`${polygon.operation}-${index}`}
                    className={polygon.operation}
                    points={polygon.points
                      .map((point) => `${point.x * imageSize.width},${point.y * imageSize.height}`)
                      .join(' ')}
                  />
                ))}
                {currentPoints.length > 0 && (
                  <>
                    <polyline
                      className={`current ${operation}`}
                      points={currentPoints
                        .map((point) => `${point.x * imageSize.width},${point.y * imageSize.height}`)
                        .join(' ')}
                    />
                    {currentPoints.map((point, index) => (
                      <circle
                        key={index}
                        className={`vertex ${operation}`}
                        cx={point.x * imageSize.width}
                        cy={point.y * imageSize.height}
                        r={Math.max(3, imageSize.width / 300)}
                      />
                    ))}
                  </>
                )}
              </svg>
            )}
          </div>
        ) : (
          <div className="viewer-empty">
            <Focus size={28} />
            <p>Исходное изображение недоступно</p>
          </div>
        )}
      </div>
      <p className="viewer-hint">
        {editing
          ? 'Клики по изображению — вершины полигона · затем «Замкнуть»'
          : 'Колесо — масштаб · перетаскивание — панорама'}
      </p>
    </section>
  );
}
