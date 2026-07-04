import {
  AbsoluteFill,
  Easing,
  interpolate,
  useCurrentFrame,
  useVideoConfig,
} from 'remotion';

const VEINS = [
  'M-80 610 C 190 390, 290 760, 560 480 S 980 220, 1280 520 S 1640 660, 1790 390',
  'M-100 260 C 220 470, 390 130, 700 330 S 1110 640, 1510 230 S 1710 110, 1820 270',
  'M220 -100 C 370 210, 690 80, 760 390 S 610 760, 940 920',
];

export function AmbientMineral() {
  const frame = useCurrentFrame();
  const { fps } = useVideoConfig();
  const cycle = (frame % (14 * fps)) / (14 * fps);
  const breathe = Math.sin(cycle * Math.PI * 2);
  const reveal = interpolate(frame, [0, 1.6 * fps], [0, 1], {
    easing: Easing.bezier(0.2, 0, 0, 1),
    extrapolateLeft: 'clamp',
    extrapolateRight: 'clamp',
  });

  return (
    <AbsoluteFill style={{ backgroundColor: '#171b18', overflow: 'hidden' }}>
      <svg
        viewBox="0 0 1728 972"
        preserveAspectRatio="xMidYMid slice"
        style={{ width: '100%', height: '100%' }}
        aria-hidden="true"
      >
        <defs>
          <filter id="grain">
            <feTurbulence
              type="fractalNoise"
              baseFrequency="0.008"
              numOctaves="4"
              seed="23"
            />
            <feColorMatrix values=".5 0 0 0 .03 0 .55 0 0 .05 0 0 .44 0 .03 0 0 0 .42 0" />
          </filter>
          <filter id="soft">
            <feGaussianBlur stdDeviation="34" />
          </filter>
          <radialGradient id="field" cx="67%" cy="38%">
            <stop offset="0" stopColor="#737263" />
            <stop offset=".42" stopColor="#343a32" />
            <stop offset="1" stopColor="#171b18" />
          </radialGradient>
          <linearGradient id="oreLine">
            <stop stopColor="#f1e9d1" stopOpacity=".08" />
            <stop offset=".48" stopColor="#d26e4a" stopOpacity=".58" />
            <stop offset="1" stopColor="#f1e9d1" stopOpacity=".04" />
          </linearGradient>
        </defs>
        <rect width="1728" height="972" fill="url(#field)" />
        <g
          opacity={0.36 * reveal}
          transform={`translate(${breathe * 7} ${breathe * -4}) scale(${1 + breathe * 0.006})`}
        >
          <ellipse cx="1190" cy="360" rx="360" ry="260" fill="#9b9782" filter="url(#soft)" opacity=".32" />
          <ellipse cx="850" cy="720" rx="310" ry="200" fill="#cc6542" filter="url(#soft)" opacity=".13" />
        </g>
        <rect width="1728" height="972" filter="url(#grain)" opacity=".94" />
        <g fill="#e9e0c8" opacity={0.08 + reveal * 0.18}>
          <path d="M1074 168l120 18 84 97-42 126-145 31-91-104z" />
          <path d="M744 536l97-43 122 61 14 133-114 73-134-61z" />
          <path d="M1372 588l100-7 75 84-30 102-124 25-67-92z" />
        </g>
        <g fill="none" stroke="url(#oreLine)" strokeWidth="3">
          {VEINS.map((path, index) => {
            const progress = (cycle + index * 0.24) % 1;
            return (
              <path
                key={path}
                d={path}
                pathLength="1"
                strokeDasharray=".22 .78"
                strokeDashoffset={-progress}
                opacity={0.45 + index * 0.15}
              />
            );
          })}
        </g>
        <g opacity={0.55 * reveal}>
          {Array.from({ length: 20 }, (_, index) => {
            const x = 800 + ((index * 193) % 780);
            const y = 90 + ((index * 127) % 760);
            const drift = Math.sin(cycle * Math.PI * 2 + index) * 9;
            return <circle key={index} cx={x + drift} cy={y - drift * 0.4} r={2 + (index % 4)} fill="#d7cdb4" />;
          })}
        </g>
      </svg>
    </AbsoluteFill>
  );
}
