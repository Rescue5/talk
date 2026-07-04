import { useEffect, useMemo, useRef, useState, type InputHTMLAttributes } from 'react';
import { Player } from '@remotion/player';
import {
  ArrowRight,
  Check,
  Clock3,
  FolderOpen,
  ImagePlus,
  Layers3,
  ShieldCheck,
  UploadCloud,
  X,
} from 'lucide-react';
import { AmbientMineral } from './AmbientMineral';
import { appendJobImages, createJob, getHistory, getJob, getResults, health, patchImageSettings, type HealthState } from './api';
import { demoResults } from './demo';
import { ProgressPanel } from './ProgressPanel';
import { errorMessage, type HistoryItem, type Job, type JobResults, type JobSettings } from './types';
import { Workspace } from './Workspace';

const ACCEPTED_EXTENSIONS = ['.jpg', '.jpeg', '.png', '.bmp', '.tif', '.tiff'];
const terminal = new Set(['completed', 'partial_failed', 'failed', 'model_unavailable']);
const imageTerminal = new Set(['completed', 'failed', 'model_unavailable']);

type DroppedEntry = {
  isFile: boolean;
  isDirectory: boolean;
  file?: (success: (file: File) => void, error?: (reason: DOMException) => void) => void;
  createReader?: () => {
    readEntries: (success: (entries: DroppedEntry[]) => void, error?: (reason: DOMException) => void) => void;
  };
};

async function walkEntry(entry: DroppedEntry): Promise<File[]> {
  if (entry.isFile && entry.file) {
    return new Promise((resolve, reject) => entry.file?.((file) => resolve([file]), reject));
  }
  if (!entry.isDirectory || !entry.createReader) return [];
  const reader = entry.createReader();
  const entries: DroppedEntry[] = [];
  while (true) {
    const batch = await new Promise<DroppedEntry[]>((resolve, reject) => reader.readEntries(resolve, reject));
    if (!batch.length) break;
    entries.push(...batch);
  }
  return (await Promise.all(entries.map(walkEntry))).flat();
}

async function filesFromDrop(dataTransfer: DataTransfer): Promise<File[]> {
  const entries = Array.from(dataTransfer.items)
    .map((item) =>
      (
        item as unknown as {
          webkitGetAsEntry?: () => DroppedEntry | null;
        }
      ).webkitGetAsEntry?.() ?? null,
    )
    .filter((entry): entry is DroppedEntry => Boolean(entry));
  if (!entries.length) return Array.from(dataTransfer.files);
  return (await Promise.all(entries.map(walkEntry))).flat();
}

function uniqueFiles(files: File[]) {
  const seen = new Set<string>();
  return files.filter((file) => {
    const lowerName = file.name.toLowerCase();
    if (!ACCEPTED_EXTENSIONS.some((extension) => lowerName.endsWith(extension))) return false;
    const key = `${file.webkitRelativePath || file.name}:${file.size}:${file.lastModified}`;
    if (seen.has(key)) return false;
    seen.add(key);
    return true;
  });
}

export function App() {
  const [files, setFiles] = useState<File[]>([]);
  const [dragging, setDragging] = useState(false);
  const [service, setService] = useState<HealthState | null>(null);
  const [job, setJob] = useState<Job | null>(null);
  const [uploadProgress, setUploadProgress] = useState<number | null>(null);
  const [results, setResults] = useState<JobResults | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [reducedMotion, setReducedMotion] = useState(false);
  const [historyEntries, setHistoryEntries] = useState<HistoryItem[]>([]);
  const [scrollY, setScrollY] = useState(0);
  const [requestedImageId, setRequestedImageId] = useState<string | null>(
    () => new URLSearchParams(window.location.search).get('image'),
  );
  const [settings, setSettings] = useState<JobSettings>({
    mode: 'overlap',
    talc_threshold_percent: 10,
    sulfide_threshold: 0.5,
    segmentation_threshold: 0.5,
    cv_threshold: 0.55,
  });
  const fileInput = useRef<HTMLInputElement>(null);
  const folderInput = useRef<HTMLInputElement>(null);

  useEffect(() => {
    const query = window.matchMedia('(prefers-reduced-motion: reduce)');
    const sync = () => setReducedMotion(query.matches);
    sync();
    query.addEventListener('change', sync);
    return () => query.removeEventListener('change', sync);
  }, []);

  useEffect(() => {
    const controller = new AbortController();
    void health(controller.signal).then(setService);
    void getHistory().then(setHistoryEntries).catch(() => setHistoryEntries([]));
    return () => controller.abort();
  }, []);

  useEffect(() => {
    if (reducedMotion) return;
    let frame = 0;
    const syncScroll = () => {
      if (frame) return;
      frame = window.requestAnimationFrame(() => {
        setScrollY(Math.min(window.scrollY, window.innerHeight));
        frame = 0;
      });
    };
    syncScroll();
    window.addEventListener('scroll', syncScroll, { passive: true });
    return () => {
      window.removeEventListener('scroll', syncScroll);
      if (frame) window.cancelAnimationFrame(frame);
    };
  }, [reducedMotion]);

  const openJob = async (id: string) => {
    setError(null);
    try {
      const restored = await getJob(id);
      setJob(restored);
      setSettings(restored.settings);
      const imageId = new URLSearchParams(window.location.search).get('image');
      const query = new URLSearchParams({ job: id });
      if (imageId) query.set('image', imageId);
      window.history.replaceState(null, '', `?${query.toString()}`);
      if (restored.status === 'completed' || restored.status === 'partial_failed') {
        setResults(await getResults(id));
      } else if (restored.status === 'running' || restored.status === 'queued') {
        const partial = await getResults(id);
        if (partial.items.some((item) => imageTerminal.has(item.status))) setResults(partial);
      } else if (restored.status === 'failed' || restored.status === 'model_unavailable') {
        setError(errorMessage(restored.error, 'Задача завершилась с ошибкой.'));
      }
    } catch (reason) {
      setError(reason instanceof Error ? reason.message : 'Задача не найдена.');
    }
  };

  useEffect(() => {
    const id = new URLSearchParams(window.location.search).get('job');
    if (id) void openJob(id);
    // The entry URL is restored once; subsequent navigation is controlled by the app.
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, []);

  useEffect(() => {
    if (!job || terminal.has(job.status)) return;
    const controller = new AbortController();
    const timer = window.setTimeout(async () => {
      try {
        const fresh = await getJob(job.id, controller.signal);
        setJob(fresh);
        const partialResults = await getResults(fresh.id);
        if (partialResults.items.some((item) => imageTerminal.has(item.status))) {
          setResults(partialResults);
        }
        if (fresh.status === 'completed' || fresh.status === 'partial_failed') {
          setResults(partialResults);
          void getHistory().then(setHistoryEntries).catch(() => undefined);
        } else if (fresh.status === 'failed' || fresh.status === 'model_unavailable') {
          setError(errorMessage(fresh.error, 'Не удалось обработать изображения.'));
        }
      } catch (reason) {
        if (!controller.signal.aborted) {
          setError(reason instanceof Error ? reason.message : 'Потеряно соединение с backend.');
        }
      }
    }, 900);
    return () => {
      controller.abort();
      clearTimeout(timer);
    };
  }, [job]);

  const totalSize = useMemo(
    () => files.reduce((sum, file) => sum + file.size, 0) / 1024 / 1024,
    [files],
  );

  const addFiles = (list: ArrayLike<File> | null) => {
    if (!list) return;
    setFiles((current) => uniqueFiles([...current, ...Array.from(list)]));
    setError(null);
  };

  const start = async () => {
    if (!files.length) return;
    setError(null);
    setUploadProgress(0);
    try {
      const created = await createJob(files, settings, setUploadProgress);
      setJob(created);
      setUploadProgress(null);
      setRequestedImageId(null);
      window.history.pushState(null, '', `?job=${encodeURIComponent(created.id)}`);
      void getHistory().then(setHistoryEntries).catch(() => undefined);
      setService((current) => current ? { ...current, reachable: true } : current);
    } catch (reason) {
      setUploadProgress(null);
      void health().then(setService);
      setError(reason instanceof Error ? reason.message : 'Не удалось создать задачу.');
    }
  };

  const reset = (nextSettings?: JobSettings) => {
    setJob(null);
    setUploadProgress(null);
    setResults(null);
    setError(null);
    setFiles([]);
    if (nextSettings) setSettings(nextSettings);
    setRequestedImageId(null);
    window.history.pushState(null, '', window.location.pathname);
    void getHistory().then(setHistoryEntries).catch(() => undefined);
  };

  if (results) {
    return (
      <Workspace
        results={results}
        settings={settings}
        initialImageId={requestedImageId}
        onImageChange={(imageId) => {
          setRequestedImageId(imageId);
          const query = new URLSearchParams({ job: results.job_id, image: imageId });
          window.history.replaceState(null, '', `?${query.toString()}`);
        }}
        onReset={reset}
        job={job}
        onAppend={async (nextFiles, sourceSettings) => {
          const updated = await appendJobImages(results.job_id, nextFiles, sourceSettings);
          setFiles((current) => uniqueFiles([...current, ...nextFiles]));
          setJob(updated);
        }}
        onPatchSettings={async (imageId, next) => {
          const updated = await patchImageSettings(results.job_id, imageId, next);
          setSettings(updated.settings ?? next);
          setJob(updated);
        }}
      />
    );
  }

  return (
    <main className="landing">
      <div className="ambient" style={{ transform: `translate3d(0, ${scrollY * 0.08}px, 0) scale(1.02)` }}>
        <Player
          component={AmbientMineral}
          compositionWidth={1728}
          compositionHeight={972}
          durationInFrames={420}
          fps={30}
          autoPlay={!reducedMotion}
          loop
          controls={false}
          initiallyMuted
          inputProps={{}}
          style={{ width: '100%', height: '100%' }}
        />
      </div>
      <div className="parallax-strata" aria-hidden="true" style={{ transform: `translate3d(0, ${scrollY * 0.16}px, 0)` }} />
      <div className="landing-vignette" aria-hidden="true" style={{ transform: `translate3d(0, ${scrollY * 0.24}px, 0)` }} />
      <header className="landing-header">
        <a className="brand" href="/" aria-label="PyTorchi: Ore analyzer — главная">
          <span className="brand-mark"><Layers3 size={18} /></span>
          <strong>PyTorchi: Ore analyzer</strong>
        </a>
        <div className={`health ${service && !service.ready ? 'offline' : ''}`}>
          <i />
          {service === null
            ? 'Проверка сервиса'
            : !service.reachable
              ? 'Сервис недоступен'
              : service.demo
                ? 'Демо backend'
                : service.ready
                  ? 'Модели готовы'
                  : 'Модели не настроены'}
        </div>
      </header>

      <section className="landing-content">
        <div className="hero-copy" style={{ translate: `0 ${scrollY * -0.05}px` }}>
          <span className="eyebrow">КОМПЬЮТЕРНОЕ ЗРЕНИЕ · МИНЕРАЛОГИЯ</span>
          <h1>Состав руды.<br /><em>В деталях.</em></h1>
          <p>Загрузите микроскопические снимки — система выделит тальк и при его доле до 10% определит рядовой или труднообогатимый класс руды.</p>
        </div>

        <div className="upload-column">
          {!job && uploadProgress === null ? (
            <>
              <section
                className={`drop-zone ${dragging ? 'dragging' : ''}`}
                onDragEnter={(event) => {
                  event.preventDefault();
                  setDragging(true);
                }}
                onDragOver={(event) => event.preventDefault()}
                onDragLeave={(event) => {
                  if (!event.currentTarget.contains(event.relatedTarget as Node)) setDragging(false);
                }}
                onDrop={async (event) => {
                  event.preventDefault();
                  setDragging(false);
                  addFiles(await filesFromDrop(event.dataTransfer));
                }}
              >
                <UploadCloud size={32} strokeWidth={1.35} />
                <h2>{files.length ? `${files.length} изображений выбрано` : 'Перетащите снимки сюда'}</h2>
                <p>{files.length ? `${totalSize.toFixed(1)} МБ · можно добавить ещё` : 'JPG, PNG, BMP, TIFF · отдельные файлы или папка'}</p>
                <div className="upload-buttons">
                  <button className="primary-button" type="button" onClick={() => fileInput.current?.click()}>
                    <ImagePlus size={17} /> Выбрать файлы
                  </button>
                  <button className="glass-button" type="button" onClick={() => folderInput.current?.click()}>
                    <FolderOpen size={17} /> Выбрать папку
                  </button>
                </div>
                <input ref={fileInput} hidden type="file" accept="image/*,.tif,.tiff" multiple onChange={(event) => addFiles(event.target.files)} />
                <input ref={folderInput} hidden type="file" multiple {...({ webkitdirectory: '', directory: '' } as InputHTMLAttributes<HTMLInputElement>)} onChange={(event) => addFiles(event.target.files)} />
              </section>

              {files.length > 0 && (
                <section className="selection-panel">
                  <div className="selection-heading">
                    <span>Очередь · {files.length}</span>
                    <button type="button" onClick={() => setFiles([])}>Очистить</button>
                  </div>
                  <div className="file-preview">
                    {files.slice(0, 4).map((file) => (
                      <div key={`${file.name}-${file.size}`}>
                        <span>{file.name}</span>
                        <button type="button" aria-label={`Убрать ${file.name}`} onClick={() => setFiles((current) => current.filter((item) => item !== file))}><X size={14} /></button>
                      </div>
                    ))}
                    {files.length > 4 && <small>+ ещё {files.length - 4}</small>}
                  </div>
                  <div className="launch-row">
                    <label>
                      Режим
                      <select value={settings.mode} onChange={(event) => setSettings((current) => ({ ...current, mode: event.target.value as JobSettings['mode'] }))}>
                        <option value="overlap">С перекрытием</option>
                        <option value="no_overlap">Без перекрытия</option>
                      </select>
                    </label>
                    <button className="launch-button" type="button" onClick={start}>
                      Начать анализ <ArrowRight size={17} />
                    </button>
                  </div>
                  <details className="advanced-settings">
                    <summary>Параметры моделей</summary>
                    <label>
                      Порог сегментации
                      <span>{settings.segmentation_threshold.toFixed(2)}</span>
                      <input
                        type="range"
                        min="0.05"
                        max="0.95"
                        step="0.05"
                        value={settings.segmentation_threshold}
                        onChange={(event) => setSettings((current) => ({ ...current, segmentation_threshold: Number(event.target.value) }))}
                      />
                    </label>
                    <label>
                      CV-уточнение
                      <span>{settings.cv_threshold.toFixed(2)}</span>
                      <input
                        type="range"
                        min="0.05"
                        max="0.95"
                        step="0.05"
                        value={settings.cv_threshold}
                        onChange={(event) => setSettings((current) => ({ ...current, cv_threshold: Number(event.target.value) }))}
                      />
                    </label>
                    <label>
                      Сульфидный классификатор
                      <span>{settings.sulfide_threshold.toFixed(2)}</span>
                      <input
                        type="range"
                        min="0.05"
                        max="0.95"
                        step="0.05"
                        value={settings.sulfide_threshold}
                        onChange={(event) => setSettings((current) => ({ ...current, sulfide_threshold: Number(event.target.value) }))}
                      />
                    </label>
                  </details>
                </section>
              )}
            </>
          ) : job ? (
            <ProgressPanel job={job} files={files} />
          ) : (
            <ProgressPanel
              files={files}
              job={{
                id: 'upload',
                status: 'queued',
                settings,
                progress: {
                  percent: (uploadProgress ?? 0) * 0.1,
                  stage: 'upload',
                  completed_images: 0,
                  total_images: files.length,
                  message: 'Загрузка изображений',
                },
              }}
            />
          )}

          {error && (
            <div className="landing-error" role="alert">
              <strong>Не удалось продолжить</strong>
              <p>{error}</p>
              <div>
                <button type="button" className="glass-button" onClick={() => setJob(null)}>Назад</button>
                <button type="button" className="demo-button" onClick={() => setResults(demoResults)}>Открыть явно помеченное демо</button>
              </div>
            </div>
          )}
          {!job && !error && (
            <button className="demo-link" type="button" onClick={() => setResults(demoResults)}>
              Нет снимков? Открыть помеченный демо-пример
            </button>
          )}
        </div>
      </section>

      <footer className="landing-footer">
        <span>v0.2 · TALC + ORE CLASSIFIER</span>
        <span>Изображения обрабатываются последовательно</span>
      </footer>
      {historyEntries.length > 0 && (
        <section className="history-section" aria-labelledby="history-title">
          <div className="history-intro">
            <span className="history-kicker"><Clock3 size={14} /> История сервиса</span>
            <h2 id="history-title">Последние анализы</h2>
            <p>Состояние и артефакты восстановятся по ссылке на задачу.</p>
          </div>
          <div className="history-list">
            {historyEntries.map((entry) => (
              <button
                key={`${entry.job_id}:${entry.image_id}`}
                type="button"
                onClick={() => {
                  setRequestedImageId(entry.image_id);
                  const query = new URLSearchParams({ job: entry.job_id, image: entry.image_id });
                  window.history.pushState(null, '', `?${query.toString()}`);
                  void openJob(entry.job_id);
                }}
              >
                <span className="history-thumb">
                  {entry.artifacts.original
                    ? <img src={entry.artifacts.original} alt="" loading="lazy" />
                    : <Layers3 size={16} />}
                </span>
                <strong>{entry.filename}</strong>
                <small>
                  {entry.classification?.label_ru ?? entry.status}
                  {' · '}
                  {new Date(entry.updated_at).toLocaleString('ru-RU')}
                  {' · '}
                  тальк {Number(entry.talc?.talc_percent ?? entry.talc?.refined_percent ?? 0).toFixed(1)}%
                </small>
                <ArrowRight size={16} />
              </button>
            ))}
          </div>
        </section>
      )}
    </main>
  );
}
