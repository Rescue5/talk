import { useEffect, useMemo, useRef, useState, type CSSProperties } from 'react';
import {
  AlertTriangle,
  ArrowLeft,
  ChevronLeft,
  ChevronRight,
  CircleGauge,
  FlaskConical,
  Layers3,
  RotateCcw,
  UploadCloud,
} from 'lucide-react';
import {
  errorMessage,
  sulfideStats,
  talcPercent as getTalcPercent,
  totalSeconds,
  type Job,
  type JobResults,
  type JobSettings,
  type ResultItem,
} from './types';
import { CompositionChart, ThresholdChart, TimingChart } from './Charts';
import { ImageViewer, type OverlayState } from './ImageViewer';

const classLabels: Record<string, string> = {
  talc_bearing: 'Оталькованная',
  ordinary: 'Рядовая',
  difficult: 'Труднообогатимая',
};
const statusLabels: Record<string, string> = {
  queued: 'В очереди',
  running: 'Обработка',
  reprocessing: 'Пересчёт',
  completed: 'Готово',
  failed: 'Ошибка',
  model_unavailable: 'Модель недоступна',
};

function Toggle({
  checked,
  onChange,
  label,
}: {
  checked: boolean;
  onChange: (value: boolean) => void;
  label: string;
}) {
  return (
    <label className="toggle-row">
      <span>{label}</span>
      <input type="checkbox" checked={checked} onChange={(event) => onChange(event.target.checked)} />
      <i aria-hidden="true" />
    </label>
  );
}

function Range({
  label,
  value,
  min,
  max,
  step,
  suffix,
  progress,
  onChange,
}: {
  label: string;
  value: number;
  min: number;
  max: number;
  step: number;
  suffix: string;
  progress?: number;
  onChange: (value: number) => void;
}) {
  return (
    <label className="range-control">
      <span>
        {label}
        <b>
          {progress != null ? `${Math.round(progress)}% · ` : ''}
          {value.toFixed(step < 1 ? 2 : 0)}{suffix}
        </b>
      </span>
      <input
        type="range"
        min={min}
        max={max}
        step={step}
        value={value}
        onChange={(event) => onChange(Number(event.target.value))}
      />
    </label>
  );
}

export function Workspace({
  job,
  results,
  settings,
  initialImageId,
  onImageChange,
  onAppend,
  onPatchSettings,
  onReset,
}: {
  job: Job | null;
  results: JobResults;
  settings: JobSettings;
  initialImageId?: string | null;
  onImageChange: (imageId: string) => void;
  onAppend: (files: File[], settings: JobSettings) => Promise<void>;
  onPatchSettings: (imageId: string, settings: JobSettings) => Promise<void>;
  onReset: (settings?: JobSettings) => void;
}) {
  const items = useMemo<ResultItem[]>(() => {
    if (!job?.images?.length) return results.items;
    const completed = new Map(results.items.map((entry) => [entry.image_id, entry]));
    return job.images.map((image) => {
      const result = completed.get(image.image_id);
      if (result) {
        return {
          ...result,
          status: image.status,
          settings: image.settings,
          progress: image.progress,
        };
      }
      return {
        image_id: image.image_id,
        filename: image.filename,
        status: image.status,
        settings: image.settings,
        progress: image.progress,
        classification: null,
        talc: null,
        sulfide: null,
        sulfide_segmentation: null,
        artifacts: {},
      };
    });
  }, [job, results]);
  const [activeIndex, setActiveIndexState] = useState(() => {
    const index = items.findIndex((entry) => entry.image_id === initialImageId);
    return index >= 0 ? index : 0;
  });
  const [inspectorWidth, setInspectorWidth] = useState(() => {
    const stored = Number(localStorage.getItem('pytorchi.inspector-width'));
    return Number.isFinite(stored) && stored >= 280 && stored <= 520 ? stored : 330;
  });
  const [settingsError, setSettingsError] = useState<string | null>(null);
  const appendInput = useRef<HTMLInputElement>(null);
  const [controls, setControls] = useState({
    talcThreshold: settings.talc_threshold_percent,
    segmentationThreshold: settings.segmentation_threshold,
    sulfideThreshold: settings.sulfide_threshold,
    cvThreshold: settings.cv_threshold,
  });
  const [overlays, setOverlays] = useState<OverlayState>({
    talc: true,
    sulfide: true,
    coarse: false,
    sulfideMaskType: 'sam',
    talcOpacity: 0.88,
    sulfideOpacity: 0.62,
    coarseOpacity: 0.55,
  });
  const item = items[activeIndex];
  const itemSettings = item?.settings ?? settings;
  const successful = useMemo(
    () => items.filter((entry) => entry.status === 'completed').length,
    [items],
  );
  useEffect(() => {
    setControls({
      talcThreshold: itemSettings.talc_threshold_percent,
      segmentationThreshold: itemSettings.segmentation_threshold,
      sulfideThreshold: itemSettings.sulfide_threshold,
      cvThreshold: itemSettings.cv_threshold,
    });
  }, [
    item?.image_id,
    itemSettings.cv_threshold,
    itemSettings.segmentation_threshold,
    itemSettings.sulfide_threshold,
    itemSettings.talc_threshold_percent,
  ]);
  const setActiveIndex = (index: number | ((current: number) => number)) => {
    setActiveIndexState((current) => {
      const next = typeof index === 'function' ? index(current) : index;
      const clamped = Math.max(0, Math.min(items.length - 1, next));
      const selected = items[clamped];
      if (selected) onImageChange(selected.image_id);
      return clamped;
    });
  };

  if (!item) {
    return (
      <main className="empty-results">
        <h1>Нет результатов</h1>
        <p>Backend завершил задачу без доступных изображений.</p>
        <button className="primary-button" onClick={() => onReset()}>Новый анализ</button>
      </main>
    );
  }

  const updateOverlay = <K extends keyof OverlayState>(key: K, value: OverlayState[K]) =>
    setOverlays((current) => ({ ...current, [key]: value }));
  const classCode = item.classification?.code ?? '';
  const className = classLabels[classCode] ?? item.classification?.label_ru ?? item.classification?.label ?? 'Не определён';
  const talcPercent = getTalcPercent(item);
  const sulfideProbability = Number(item.classification?.confidence ?? 0);
  const hasSulfideCv = Boolean(item.artifacts.sulfide_cv_overlay || item.artifacts.sulfide_mask);
  const hasSulfideSam = Boolean(item.artifacts.sulfide_sam_overlay);
  const hasSulfideMask = hasSulfideCv || hasSulfideSam;
  const effectiveSulfideMaskType = overlays.sulfideMaskType === 'sam'
    ? (hasSulfideSam ? 'sam' : 'cv')
    : (hasSulfideCv ? 'cv' : hasSulfideSam ? 'sam' : 'cv');
  const sulfideComposition = sulfideStats(item, effectiveSulfideMaskType);
  const sulfidePercent = Number(sulfideComposition?.percent ?? 0);
  const sulfideComponents = Number(sulfideComposition?.component_count ?? 0);
  const reprocessStage = item.status === 'running' || item.status === 'reprocessing'
    ? item.progress?.stage ?? ''
    : '';
  const reprocessPercent = item.progress?.percent;

  return (
    <div className="workspace-page">
      <header className="workspace-header">
        <button className="brand-button" type="button" onClick={() => onReset()}>
          <span className="brand-mark"><Layers3 size={17} /></span>
          <span>PyTorchi: Ore analyzer</span>
        </button>
        <div className="workspace-title">
          <span>{results.demo ? 'ДЕМО · ' : ''}ЗАДАЧА {results.job_id.slice(0, 8)}</span>
          <strong>{item.filename}</strong>
        </div>
        <div className="workspace-actions">
          <span>{successful}/{items.length} готово</span>
          <button type="button" className="secondary-button" onClick={() => onReset()}>
            <ArrowLeft size={15} /> Новый анализ
          </button>
        </div>
      </header>

      {results.demo && (
        <div className="demo-banner" role="status">
          Демо-режим: показаны детерминированные примеры, а не результаты модели.
        </div>
      )}

      <main
        className="workspace-grid"
        style={{ '--inspector-width': `${inspectorWidth}px` } as CSSProperties}
      >
        <aside
          className="image-rail"
          aria-label="Изображения задачи"
          onDragOver={(event) => event.preventDefault()}
          onDrop={(event) => {
            event.preventDefault();
            const next = Array.from(event.dataTransfer.files).filter((file) =>
              /\.(jpe?g|png|bmp|tiff?)$/i.test(file.name),
            );
            if (next.length) void onAppend(next, itemSettings);
          }}
        >
          <div className="rail-heading">
            <span>Снимки</span>
            <b>{items.length}</b>
          </div>
          <div className="image-list">
            {items.map((entry, index) => (
              <button
                key={entry.image_id}
                type="button"
                className={index === activeIndex ? 'active' : ''}
                onClick={() => setActiveIndex(index)}
                aria-current={index === activeIndex}
              >
                <span>{String(index + 1).padStart(2, '0')}</span>
                <div>
                  <strong>{entry.filename}</strong>
                  <small>
                    {classLabels[entry.classification?.code ?? ''] ?? entry.classification?.label_ru ?? statusLabels[entry.status] ?? entry.status}
                    {entry.progress?.percent != null
                      ? ` · ${Math.round(entry.progress.percent)}%`
                      : ''}
                  </small>
                  {(entry.status === 'queued' || entry.status === 'running' || entry.status === 'reprocessing') && (
                    <i className="rail-progress" aria-hidden="true">
                      <b style={{ width: `${Math.max(2, entry.progress?.percent ?? 0)}%` }} />
                    </i>
                  )}
                </div>
              </button>
            ))}
          </div>
          <button className="append-button" type="button" onClick={() => appendInput.current?.click()}>
            <UploadCloud size={14} /> Добавить снимки
          </button>
          <input
            ref={appendInput}
            hidden
            type="file"
            multiple
            accept=".jpg,.jpeg,.png,.bmp,.tif,.tiff"
            onChange={(event) => {
              const next = Array.from(event.target.files ?? []);
              if (next.length) void onAppend(next, itemSettings);
              event.target.value = '';
            }}
          />
        </aside>

        <section className="visual-column">
          <div className="visual-meta">
            <div>
              <span className={`classification ${classCode}`}>{className}</span>
              <span className="confidence">
                {classCode === 'talc_bearing'
                  ? `Тальк ${talcPercent.toFixed(1)}% · порог ${itemSettings.talc_threshold_percent.toFixed(0)}%`
                  : (classCode === 'ordinary' || classCode === 'difficult') && item.classification?.confidence != null
                    ? `Достоверность классификатора ${Math.round(item.classification.confidence * 100)}%`
                    : item.progress?.message ?? statusLabels[item.status] ?? item.status}
              </span>
            </div>
            <div className="image-stepper">
              <button
                type="button"
                aria-label="Предыдущее изображение"
                disabled={activeIndex === 0}
                onClick={() => setActiveIndex((index) => Math.max(0, index - 1))}
              >
                <ChevronLeft size={17} />
              </button>
              <span>{activeIndex + 1} / {items.length}</span>
              <button
                type="button"
                aria-label="Следующее изображение"
                disabled={activeIndex === items.length - 1}
                onClick={() => setActiveIndex((index) => Math.min(items.length - 1, index + 1))}
              >
                <ChevronRight size={17} />
              </button>
            </div>
          </div>
          <div className="visual-alerts">
            {item.error && (
              <div className="result-error" role="alert">
                {errorMessage(item.error, 'Изображение не обработано')}
              </div>
            )}
            {item.warnings?.map((warning) => (
              <div className="quality-warning" role="status" key={warning.code}>
                <AlertTriangle size={16} />
                <span>{warning.message}</span>
              </div>
            ))}
          </div>
          <ImageViewer
            artifacts={item.artifacts}
            overlays={{ ...overlays, sulfideMaskType: effectiveSulfideMaskType }}
            filename={item.filename}
          />
          <div className="summary-strip">
            <div><span>Тальк</span><strong>{talcPercent.toFixed(1)}%</strong></div>
            <div>
              <span>{classCode === 'ordinary' || classCode === 'difficult' ? 'Достоверность классификатора' : 'Порог талька'}</span>
              <strong>
                {classCode === 'ordinary' || classCode === 'difficult'
                  ? sulfideProbability > 0 ? `${(sulfideProbability * 100).toFixed(0)}%` : '—'
                  : `${itemSettings.talc_threshold_percent.toFixed(0)}%`}
              </strong>
            </div>
            <div><span>Сульфиды · {sulfideComponents} зон</span><strong>{sulfidePercent.toFixed(1)}%</strong></div>
            <div><span>Общее время</span><strong>{totalSeconds(item).toFixed(1)} c</strong></div>
          </div>
        </section>

        <aside className="inspector">
          <button
            className="inspector-resizer"
            type="button"
            role="separator"
            aria-orientation="vertical"
            aria-valuemin={280}
            aria-valuemax={520}
            aria-valuenow={inspectorWidth}
            aria-label="Изменить ширину панели"
            onKeyDown={(event) => {
              if (!['ArrowLeft', 'ArrowRight', 'Home', 'End'].includes(event.key)) return;
              event.preventDefault();
              const next = event.key === 'Home'
                ? 280
                : event.key === 'End'
                  ? 520
                  : Math.max(280, Math.min(520, inspectorWidth + (event.key === 'ArrowLeft' ? 16 : -16)));
              setInspectorWidth(next);
              localStorage.setItem('pytorchi.inspector-width', String(next));
            }}
            onPointerDown={(event) => {
              event.preventDefault();
              const startX = event.clientX;
              const startWidth = inspectorWidth;
              let latestWidth = startWidth;
              const move = (moveEvent: PointerEvent) => {
                const next = Math.max(280, Math.min(520, startWidth + startX - moveEvent.clientX));
                latestWidth = next;
                setInspectorWidth(next);
              };
              const stop = () => {
                localStorage.setItem('pytorchi.inspector-width', String(latestWidth));
                window.removeEventListener('pointermove', move);
                window.removeEventListener('pointerup', stop);
              };
              window.addEventListener('pointermove', move);
              window.addEventListener('pointerup', stop);
            }}
          />
          <section className="inspector-section">
            <h2><FlaskConical size={16} /> Слои</h2>
            <Toggle label="Уточнённый тальк" checked={overlays.talc} onChange={(value) => updateOverlay('talc', value)} />
            <Range label="Прозрачность" value={overlays.talcOpacity} min={0} max={1} step={0.05} suffix="" onChange={(value) => updateOverlay('talcOpacity', value)} />
            {hasSulfideMask && (
              <>
                <Toggle label="Сульфиды" checked={overlays.sulfide} onChange={(value) => updateOverlay('sulfide', value)} />
                <div className="segmented-control" role="group" aria-label="Тип сульфидной маски">
                  <button
                    type="button"
                    className={effectiveSulfideMaskType === 'cv' ? 'active' : ''}
                    disabled={!hasSulfideCv}
                    onClick={() => updateOverlay('sulfideMaskType', 'cv')}
                  >
                    CV
                  </button>
                  <button
                    type="button"
                    className={effectiveSulfideMaskType === 'sam' ? 'active' : ''}
                    disabled={!hasSulfideSam}
                    onClick={() => updateOverlay('sulfideMaskType', 'sam')}
                  >
                    SAM
                  </button>
                </div>
                <Range label="Прозрачность" value={overlays.sulfideOpacity} min={0} max={1} step={0.05} suffix="" onChange={(value) => updateOverlay('sulfideOpacity', value)} />
              </>
            )}
            <Toggle label="Грубая маска" checked={overlays.coarse} onChange={(value) => updateOverlay('coarse', value)} />
            <Range label="Прозрачность" value={overlays.coarseOpacity} min={0} max={1} step={0.05} suffix="" onChange={(value) => updateOverlay('coarseOpacity', value)} />
          </section>

          <section className="inspector-section">
            <h2><CircleGauge size={16} /> Пороговые значения</h2>
            <Range label="Класс талька" value={controls.talcThreshold} min={1} max={30} step={1} suffix="%" progress={reprocessStage.includes('classification') ? reprocessPercent : undefined} onChange={(value) => setControls((state) => ({ ...state, talcThreshold: value }))} />
            <Range label="Сегментация" value={controls.segmentationThreshold} min={0.05} max={0.95} step={0.05} suffix="" progress={reprocessStage.includes('segmentation_threshold') ? reprocessPercent : undefined} onChange={(value) => setControls((state) => ({ ...state, segmentationThreshold: value }))} />
            <Range label="CV-уточнение" value={controls.cvThreshold} min={0.05} max={0.95} step={0.05} suffix="" progress={reprocessStage.includes('cv_refinement') ? reprocessPercent : undefined} onChange={(value) => setControls((state) => ({ ...state, cvThreshold: value }))} />
            <Range label="Сульфидный класс" value={controls.sulfideThreshold} min={0.05} max={0.95} step={0.05} suffix="" progress={reprocessStage.includes('classification') ? reprocessPercent : undefined} onChange={(value) => setControls((state) => ({ ...state, sulfideThreshold: value }))} />
            <p className="control-note">Сохранение применит параметры к выбранному снимку и запустит его повторную обработку.</p>
            <button
              className="rerun-button"
              type="button"
              onClick={async () => {
                setSettingsError(null);
                try {
                  await onPatchSettings(item.image_id, {
                    ...itemSettings,
                    talc_threshold_percent: controls.talcThreshold,
                    segmentation_threshold: controls.segmentationThreshold,
                    cv_threshold: controls.cvThreshold,
                    sulfide_threshold: controls.sulfideThreshold,
                  });
                } catch (reason) {
                  setSettingsError(reason instanceof Error ? reason.message : 'Не удалось сохранить параметры.');
                }
              }}
            >
              Сохранить для этого снимка
            </button>
            {settingsError && <p className="settings-error" role="alert">{settingsError}</p>}
            <button
              className="text-button"
              type="button"
              onClick={() => setControls({
                talcThreshold: itemSettings.talc_threshold_percent,
                segmentationThreshold: itemSettings.segmentation_threshold,
                sulfideThreshold: itemSettings.sulfide_threshold,
                cvThreshold: itemSettings.cv_threshold,
              })}
            >
              <RotateCcw size={14} /> Сбросить
            </button>
          </section>

          <section className="inspector-section charts">
            <ThresholdChart item={item} threshold={controls.talcThreshold} />
            <CompositionChart item={item} sulfideMaskType={effectiveSulfideMaskType} />
            <TimingChart item={item} />
          </section>
        </aside>
      </main>
    </div>
  );
}
