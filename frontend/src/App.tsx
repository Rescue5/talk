import {
  useEffect,
  useMemo,
  useRef,
  useState,
  type CSSProperties,
  type InputHTMLAttributes,
} from 'react';
import { Player } from '@remotion/player';
import {
  ArrowRight,
  ChevronDown,
  Clock3,
  Database,
  FolderOpen,
  ImagePlus,
  Layers3,
  Trash2,
  UploadCloud,
  X,
} from 'lucide-react';
import { AmbientMineral } from './AmbientMineral';
import {
  appendJobImages,
  clearCache,
  createJob,
  getCacheInfo,
  getHistory,
  getJob,
  getResults,
  health,
  patchImageSettings,
  updateCacheLimit,
  type HealthState,
} from './api';
import { demoResults } from './demo';
import { ProgressPanel } from './ProgressPanel';
import {
  errorMessage,
  type CacheInfo,
  type HistoryItem,
  type Job,
  type JobResults,
  type JobSettings,
} from './types';
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

type PointerState = {
  x: number;
  y: number;
};

type FloatingElement = {
  style: CSSProperties;
};

const layerBaseStyle: CSSProperties = {
  position: 'absolute',
  inset: 0,
  pointerEvents: 'none',
  willChange: 'transform',
};

const smoothTransformStyle: CSSProperties = {
  transition: 'transform 55ms linear',
};

const scrollHintStyle: CSSProperties = {
  position: 'absolute',
  left: '50%',
  bottom: 28,
  zIndex: 8,
  transform: 'translateX(-50%)',
  display: 'inline-flex',
  alignItems: 'center',
  gap: 8,
  padding: '9px 13px',
  border: '1px solid rgba(245, 237, 205, 0.24)',
  borderRadius: 999,
  color: 'rgba(245, 237, 205, 0.82)',
  background: 'linear-gradient(135deg, rgba(245, 237, 205, 0.09), rgba(218, 113, 72, 0.06))',
  boxShadow: '0 16px 50px rgba(0, 0, 0, 0.24), inset 0 1px 0 rgba(255, 255, 255, 0.08)',
  backdropFilter: 'blur(12px)',
  WebkitBackdropFilter: 'blur(12px)',
  fontSize: 12,
  letterSpacing: '0.12em',
  textTransform: 'uppercase',
  cursor: 'pointer',
};

const polygons: FloatingElement[] = [
  {
    style: {
      position: 'absolute',
      left: '61%',
      top: '7%',
      width: 260,
      height: 230,
      opacity: 0.16,
      background: 'rgba(245, 241, 218, 0.34)',
      clipPath: 'polygon(26% 0%, 76% 9%, 100% 45%, 82% 86%, 28% 100%, 0% 50%)',
      filter: 'blur(0.2px)',
    },
  },
  {
    style: {
      position: 'absolute',
      left: '41%',
      top: '53%',
      width: 290,
      height: 250,
      opacity: 0.13,
      background: 'rgba(232, 229, 204, 0.28)',
      clipPath: 'polygon(45% 0%, 93% 26%, 100% 74%, 53% 100%, 0% 77%, 7% 20%)',
      filter: 'blur(0.2px)',
    },
  },
  {
    style: {
      position: 'absolute',
      right: '6%',
      bottom: '4%',
      width: 220,
      height: 210,
      opacity: 0.17,
      background: 'rgba(236, 232, 206, 0.32)',
      clipPath: 'polygon(21% 7%, 72% 3%, 100% 48%, 83% 88%, 27% 100%, 0% 51%)',
      filter: 'blur(0.2px)',
    },
  },
];

const dots: FloatingElement[] = [
  {
    style: {
      position: 'absolute',
      left: '56%',
      top: '15%',
      width: 7,
      height: 7,
      borderRadius: 999,
      background: 'rgba(245, 237, 205, 0.36)',
      boxShadow: '26px -4px 0 rgba(245,237,205,0.28)',
    },
  },
  {
    style: {
      position: 'absolute',
      left: '70%',
      top: '32%',
      width: 9,
      height: 9,
      borderRadius: 999,
      background: 'rgba(245, 237, 205, 0.30)',
      boxShadow: '32px -6px 0 rgba(245,237,205,0.22)',
    },
  },
  {
    style: {
      position: 'absolute',
      right: '16%',
      top: '14%',
      width: 12,
      height: 12,
      borderRadius: 999,
      background: 'rgba(245, 237, 205, 0.28)',
      boxShadow: '28px -3px 0 rgba(245,237,205,0.22)',
    },
  },
  {
    style: {
      position: 'absolute',
      left: '57%',
      bottom: '13%',
      width: 7,
      height: 7,
      borderRadius: 999,
      background: 'rgba(245, 237, 205, 0.30)',
      boxShadow: '24px -2px 0 rgba(245,237,205,0.24)',
    },
  },
  {
    style: {
      position: 'absolute',
      right: '11%',
      bottom: '12%',
      width: 12,
      height: 12,
      borderRadius: 999,
      background: 'rgba(245, 237, 205, 0.26)',
    },
  },
];

const lines: FloatingElement[] = [
  {
    style: {
      position: 'absolute',
      left: '22%',
      top: '-3%',
      width: 360,
      height: 3,
      borderRadius: 999,
      background: 'linear-gradient(90deg, transparent, rgba(218, 113, 72, 0.22), transparent)',
      transform: 'rotate(26deg)',
    },
  },
  {
    style: {
      position: 'absolute',
      right: '-3%',
      top: '58%',
      width: 460,
      height: 3,
      borderRadius: 999,
      background: 'linear-gradient(90deg, transparent, rgba(218, 113, 72, 0.20), transparent)',
      transform: 'rotate(42deg)',
    },
  },
  {
    style: {
      position: 'absolute',
      right: '2%',
      bottom: '18%',
      width: 360,
      height: 2,
      borderRadius: 999,
      background: 'linear-gradient(90deg, transparent, rgba(245, 237, 205, 0.18), transparent)',
      transform: 'rotate(-10deg)',
    },
  },
];

function transformLayer(x: number, y: number, scroll: number, scale = 1): string {
  return `translate3d(${x}px, ${y + scroll}px, 0) scale(${scale})`;
}

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

function formatBytes(value: number): string {
  if (value < 1024 * 1024) return `${(value / 1024).toFixed(0)} КБ`;
  if (value < 1024 * 1024 * 1024) return `${(value / 1024 / 1024).toFixed(1)} МБ`;
  return `${(value / 1024 / 1024 / 1024).toFixed(2)} ГБ`;
}

function imageWord(value: number): string {
  const tens = value % 100;
  const units = value % 10;

  if (tens >= 11 && tens <= 14) return 'снимков';
  if (units === 1) return 'снимок';
  if (units >= 2 && units <= 4) return 'снимка';

  return 'снимков';
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
  const [cacheInfo, setCacheInfo] = useState<CacheInfo | null>(null);
  const [cacheLimit, setCacheLimit] = useState(50);
  const [cacheBusy, setCacheBusy] = useState(false);
  const [cacheError, setCacheError] = useState<string | null>(null);
  const [scrollY, setScrollY] = useState(0);
  const [pointer, setPointer] = useState<PointerState>({ x: 0, y: 0 });
  const [time, setTime] = useState(0);
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

  const parallax = useMemo(() => {
    if (reducedMotion) {
      return {
        ambientX: 0,
        ambientY: 0,
        polygonX: 0,
        polygonY: 0,
        dotsX: 0,
        dotsY: 0,
        linesX: 0,
        linesY: 0,
        strataX: 0,
        strataY: 0,
        vignetteX: 0,
        vignetteY: 0,
        copyX: 0,
        copyY: 0,
        scrollAmbient: 0,
        scrollStrata: 0,
        scrollVignette: 0,
      };
    }

    const idleAmbientX = Math.sin(time * 0.18) * 6;
    const idleAmbientY = Math.cos(time * 0.16) * 5;
    const idlePolygonX = Math.sin(time * 0.24 + 1.4) * 9;
    const idlePolygonY = Math.cos(time * 0.21 + 0.6) * 7;
    const idleDotsX = Math.sin(time * 0.38 + 2.1) * 11;
    const idleDotsY = Math.cos(time * 0.31 + 1.2) * 8;
    const idleLinesX = Math.sin(time * 1.65) * 42;
    const idleLinesY = Math.cos(time * 1.25) * 26;
    const idleStrataX = Math.sin(time * 0.22 + 0.4) * 10;
    const idleStrataY = Math.cos(time * 0.19 + 1.1) * 7;
    const idleVignetteX = Math.sin(time * 0.14 + 2.8) * 5;
    const idleVignetteY = Math.cos(time * 0.13 + 2.2) * 4;

    return {
      ambientX: pointer.x * -14 + idleAmbientX,
      ambientY: pointer.y * -10 + idleAmbientY,
      polygonX: pointer.x * -34 + idlePolygonX,
      polygonY: pointer.y * -24 + idlePolygonY,
      dotsX: pointer.x * -56 + idleDotsX,
      dotsY: pointer.y * -38 + idleDotsY,
      linesX: pointer.x * -24 + idleLinesX,
      linesY: pointer.y * -18 + idleLinesY,
      strataX: pointer.x * -44 + idleStrataX,
      strataY: pointer.y * -30 + idleStrataY,
      vignetteX: pointer.x * -18 + idleVignetteX,
      vignetteY: pointer.y * -12 + idleVignetteY,
      copyX: pointer.x * 6,
      copyY: pointer.y * 4,
      scrollAmbient: scrollY * 0.08,
      scrollStrata: scrollY * 0.16,
      scrollVignette: scrollY * 0.24,
    };
  }, [pointer.x, pointer.y, reducedMotion, scrollY, time]);

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

    void getHistory(500)
      .then(setHistoryEntries)
      .catch(() => setHistoryEntries([]));

    void getCacheInfo()
      .then((info) => {
        setCacheInfo(info);
        setCacheLimit(info.max_images);
      })
      .catch(() => setCacheInfo(null));

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

  useEffect(() => {
    if (reducedMotion) return;

    let frame = 0;
    let nextPointer: PointerState = { x: 0, y: 0 };

    const syncPointer = (event: PointerEvent) => {
      const width = Math.max(window.innerWidth, 1);
      const height = Math.max(window.innerHeight, 1);

      nextPointer = {
        x: event.clientX / width - 0.5,
        y: event.clientY / height - 0.5,
      };

      if (frame) return;

      frame = window.requestAnimationFrame(() => {
        setPointer(nextPointer);
        frame = 0;
      });
    };

    const resetPointer = () => {
      nextPointer = { x: 0, y: 0 };

      if (frame) return;

      frame = window.requestAnimationFrame(() => {
        setPointer(nextPointer);
        frame = 0;
      });
    };

    window.addEventListener('pointermove', syncPointer, { passive: true });
    window.addEventListener('pointerleave', resetPointer);

    return () => {
      window.removeEventListener('pointermove', syncPointer);
      window.removeEventListener('pointerleave', resetPointer);

      if (frame) window.cancelAnimationFrame(frame);
    };
  }, [reducedMotion]);

  useEffect(() => {
    if (reducedMotion) return;

    let frame = 0;
    const startedAt = performance.now();

    const tick = (now: number) => {
      setTime((now - startedAt) / 1000);
      frame = window.requestAnimationFrame(tick);
    };

    frame = window.requestAnimationFrame(tick);

    return () => {
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

        if (partial.items.some((item) => imageTerminal.has(item.status))) {
          setResults(partial);
        }
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

          void getHistory(500)
            .then(setHistoryEntries)
            .catch(() => undefined);

          void getCacheInfo()
            .then(setCacheInfo)
            .catch(() => undefined);
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

      void getHistory(500)
        .then(setHistoryEntries)
        .catch(() => undefined);

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

    void getHistory(500)
      .then(setHistoryEntries)
      .catch(() => undefined);
  };

  if (results) {
    return (
      <Workspace
        results={results}
        settings={settings}
        initialImageId={requestedImageId}
        onImageChange={(imageId) => {
          setRequestedImageId(imageId);

          const query = new URLSearchParams({
            job: results.job_id,
            image: imageId,
          });

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
      <section className="landing-hero">
        <div
          className="ambient"
          style={{
            ...layerBaseStyle,
            ...smoothTransformStyle,
            transform: transformLayer(
              parallax.ambientX,
              parallax.ambientY,
              parallax.scrollAmbient,
              1.08,
            ),
          }}
        >
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

        <div
          className="mineral-layer mineral-polygons"
          aria-hidden="true"
          style={{
            ...layerBaseStyle,
            ...smoothTransformStyle,
            zIndex: 1,
            transform: transformLayer(
              parallax.polygonX,
              parallax.polygonY,
              parallax.scrollStrata,
              1.02,
            ),
          }}
        >
          {polygons.map((item, index) => (
            <span key={`polygon-${index}`} style={item.style} />
          ))}
        </div>

        <div
          className="mineral-layer mineral-lines"
          aria-hidden="true"
          style={{
            ...layerBaseStyle,
            ...smoothTransformStyle,
            zIndex: 2,
            transform: transformLayer(
              parallax.linesX,
              parallax.linesY,
              parallax.scrollStrata,
              1.04,
            ),
          }}
        >
          {lines.map((item, index) => (
            <span key={`line-${index}`} style={item.style} />
          ))}
        </div>

        <div
          className="parallax-strata"
          aria-hidden="true"
          style={{
            ...smoothTransformStyle,
            transform: transformLayer(
              parallax.strataX,
              parallax.strataY,
              parallax.scrollStrata,
            ),
          }}
        />

        <div
          className="mineral-layer mineral-dots"
          aria-hidden="true"
          style={{
            ...layerBaseStyle,
            ...smoothTransformStyle,
            zIndex: 3,
            transform: transformLayer(
              parallax.dotsX,
              parallax.dotsY,
              parallax.scrollVignette,
              1.01,
            ),
          }}
        >
          {dots.map((item, index) => (
            <span key={`dot-${index}`} style={item.style} />
          ))}
        </div>

        <div
          className="landing-vignette"
          aria-hidden="true"
          style={{
            ...smoothTransformStyle,
            transform: transformLayer(
              parallax.vignetteX,
              parallax.vignetteY,
              parallax.scrollVignette,
            ),
          }}
        />

        <header className="landing-header">
          <a className="brand" href="/" aria-label="PyTorchi: Ore analyzer — главная">
            <span className="brand-mark">
              <Layers3 size={18} />
            </span>
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
          <div
            className="hero-copy"
            style={{
              ...smoothTransformStyle,
              transform: transformLayer(parallax.copyX, parallax.copyY, scrollY * -0.04),
            }}
          >
            <h1>
              Состав руды.
              <br />
              <em>В деталях.</em>
            </h1>

            <p>
              Загрузите микроскопические снимки — система выделит тальк и при его доле до
              10% определит рядовой или труднообогатимый класс руды.
            </p>
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
                    if (!event.currentTarget.contains(event.relatedTarget as Node)) {
                      setDragging(false);
                    }
                  }}
                  onDrop={async (event) => {
                    event.preventDefault();
                    setDragging(false);
                    addFiles(await filesFromDrop(event.dataTransfer));
                  }}
                >
                  <UploadCloud size={32} strokeWidth={1.35} />

                  <h2>
                    {files.length
                      ? `${files.length} изображений выбрано`
                      : 'Перетащите снимки сюда'}
                  </h2>

                  <p>
                    {files.length
                      ? `${totalSize.toFixed(1)} МБ · можно добавить ещё`
                      : 'JPG, PNG, BMP, TIFF · отдельные файлы или папка'}
                  </p>

                  <div className="upload-buttons">
                    <button
                      className="primary-button"
                      type="button"
                      onClick={() => fileInput.current?.click()}
                    >
                      <ImagePlus size={17} /> Выбрать файлы
                    </button>

                    <button
                      className="glass-button"
                      type="button"
                      onClick={() => folderInput.current?.click()}
                    >
                      <FolderOpen size={17} /> Выбрать папку
                    </button>
                  </div>

                  <input
                    ref={fileInput}
                    hidden
                    type="file"
                    accept="image/*,.tif,.tiff"
                    multiple
                    onChange={(event) => addFiles(event.target.files)}
                  />

                  <input
                    ref={folderInput}
                    hidden
                    type="file"
                    multiple
                    {...({ webkitdirectory: '', directory: '' } as InputHTMLAttributes<HTMLInputElement>)}
                    onChange={(event) => addFiles(event.target.files)}
                  />
                </section>

                {files.length > 0 && (
                  <section className="selection-panel">
                    <div className="selection-heading">
                      <span>Очередь · {files.length}</span>

                      <button type="button" onClick={() => setFiles([])}>
                        Очистить
                      </button>
                    </div>

                    <div className="file-preview">
                      {files.slice(0, 4).map((file) => (
                        <div key={`${file.name}-${file.size}`}>
                          <span>{file.name}</span>

                          <button
                            type="button"
                            aria-label={`Убрать ${file.name}`}
                            onClick={() =>
                              setFiles((current) => current.filter((item) => item !== file))
                            }
                          >
                            <X size={14} />
                          </button>
                        </div>
                      ))}

                      {files.length > 4 && <small>+ ещё {files.length - 4}</small>}
                    </div>

                    <div className="launch-row">
                      <label>
                        Режим
                        <select
                          value={settings.mode}
                          onChange={(event) =>
                            setSettings((current) => ({
                              ...current,
                              mode: event.target.value as JobSettings['mode'],
                            }))
                          }
                        >
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
                          onChange={(event) =>
                            setSettings((current) => ({
                              ...current,
                              segmentation_threshold: Number(event.target.value),
                            }))
                          }
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
                          onChange={(event) =>
                            setSettings((current) => ({
                              ...current,
                              cv_threshold: Number(event.target.value),
                            }))
                          }
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
                          onChange={(event) =>
                            setSettings((current) => ({
                              ...current,
                              sulfide_threshold: Number(event.target.value),
                            }))
                          }
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
                  <button type="button" className="glass-button" onClick={() => setJob(null)}>
                    Назад
                  </button>

                  <button
                    type="button"
                    className="demo-button"
                    onClick={() => setResults(demoResults)}
                  >
                    Открыть явно помеченное демо
                  </button>
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


        <style>
          {`
            .scroll-cue {
              animation: scroll-cue-float 3.0s ease-in-out infinite;
            }

            .scroll-cue:hover {
              border-color: rgba(245, 237, 205, 0.42);
              color: rgba(245, 237, 205, 0.96);
              background: linear-gradient(135deg, rgba(245, 237, 205, 0.13), rgba(218, 113, 72, 0.10));
            }

            .scroll-cue svg {
              filter: drop-shadow(0 0 10px rgba(218, 113, 72, 0.45));
            }

            @keyframes scroll-cue-float {
              0%, 100% {
                translate: 0 0;
                opacity: 0.72;
              }

              50% {
                translate: 0 8px;
                opacity: 1;
              }
            }

            @media (prefers-reduced-motion: reduce) {
              .scroll-cue {
                animation: none;
              }
            }
          `}
        </style>

        <button
          className="scroll-cue"
          type="button"
          style={scrollHintStyle}
          aria-label="Прокрутить к истории анализов"
          onClick={() => document.getElementById('history-title')?.scrollIntoView({ behavior: 'smooth' })}
        >
          <span>История</span>
          <ChevronDown size={17} strokeWidth={1.6} />
        </button>

        <footer className="landing-footer">
          <span>v0.2 · TALC + ORE CLASSIFIER</span>
          <span>Изображения обрабатываются последовательно</span>
        </footer>
      </section>

      <section className="history-section" aria-labelledby="history-title">
        <div className="history-intro">
          <span className="history-kicker">
            <Clock3 size={14} /> История сервиса
          </span>

          <h2 id="history-title">Последние анализы</h2>

          <p>Состояние и артефакты восстановятся по ссылке на задачу.</p>

          <div className="cache-panel">
            <div className="cache-summary">
              <Database size={16} />

              <span>
                <strong>
                  {cacheInfo?.stored_images ?? historyEntries.length}{' '}
                  {imageWord(cacheInfo?.stored_images ?? historyEntries.length)}
                </strong>
                {cacheInfo ? ` · ${formatBytes(cacheInfo.size_bytes)}` : ''}
              </span>
            </div>

            <label>
              Хранить снимков
              <input
                type="number"
                min={1}
                max={500}
                value={cacheLimit}
                onChange={(event) => setCacheLimit(Number(event.target.value))}
              />
            </label>

            <div className="cache-actions">
              <button
                type="button"
                disabled={cacheBusy || cacheLimit < 1 || cacheLimit > 500}
                onClick={async () => {
                  setCacheBusy(true);
                  setCacheError(null);

                  try {
                    const info = await updateCacheLimit(cacheLimit);

                    setCacheInfo(info);
                    setHistoryEntries(await getHistory(500));
                  } catch (reason) {
                    setCacheError(reason instanceof Error ? reason.message : 'Не удалось изменить кэш.');
                  } finally {
                    setCacheBusy(false);
                  }
                }}
              >
                Сохранить лимит
              </button>

              <button
                className="cache-clear"
                type="button"
                disabled={cacheBusy || (cacheInfo?.stored_images ?? historyEntries.length) === 0}
                onClick={async () => {
                  if (!window.confirm('Удалить сохранённые снимки и результаты анализа?')) return;

                  setCacheBusy(true);
                  setCacheError(null);

                  try {
                    const info = await clearCache();

                    setCacheInfo(info);
                    setHistoryEntries([]);
                  } catch (reason) {
                    setCacheError(reason instanceof Error ? reason.message : 'Не удалось очистить кэш.');
                  } finally {
                    setCacheBusy(false);
                  }
                }}
              >
                <Trash2 size={14} /> Очистить
              </button>
            </div>

            {cacheError && (
              <p className="cache-error" role="alert">
                {cacheError}
              </p>
            )}
          </div>
        </div>

        <div className="history-list">
          {historyEntries.length === 0 ? (
            <p className="history-empty">Здесь появятся завершённые анализы.</p>
          ) : (
            historyEntries.map((entry) => (
              <button
                key={`${entry.job_id}:${entry.image_id}`}
                type="button"
                onClick={() => {
                  setRequestedImageId(entry.image_id);

                  const query = new URLSearchParams({
                    job: entry.job_id,
                    image: entry.image_id,
                  });

                  window.history.pushState(null, '', `?${query.toString()}`);

                  void openJob(entry.job_id);
                }}
              >
                <span className="history-thumb">
                  {entry.artifacts.original ? (
                    <img src={entry.artifacts.original} alt="" loading="lazy" />
                  ) : (
                    <Layers3 size={16} />
                  )}
                </span>

                <strong>{entry.filename}</strong>

                <small>
                  {entry.classification?.label_ru ?? entry.status}
                  {' · '}
                  {new Date(entry.updated_at).toLocaleString('ru-RU')}
                  {' · '}
                  тальк{' '}
                  {Number(entry.talc?.talc_percent ?? entry.talc?.refined_percent ?? 0).toFixed(1)}%
                </small>

                <ArrowRight size={16} />
              </button>
            ))
          )}
        </div>
      </section>
    </main>
  );
}
