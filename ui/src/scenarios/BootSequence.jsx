import { useTypewriter } from '../hooks/useTypewriter';
import styles from './BootSequence.module.css';

const BOOT_LINES = [
  '> INITIALIZING ARGUS DEFENSE MATRIX...',
  '> LOADING PYDANTIC STATE ENGINE.............. OK',
  '> CONNECTING TO OPENENV VALIDATOR............ OK',
  '> SCANNING CORPORATE NETWORK (6 DEPARTMENTS). OK',
  '> LOADING 7 ESPIONAGE MECHANICS.............. OK',
  '> CALIBRATING 5-DIMENSION GRADER............. OK',
  '> ⚠ ANOMALIES DETECTED: 5 UNKNOWN SIGNATURES',
  '> THREAT LEVEL: ██████████ CRITICAL',
  '> ARGUS OS v3.0 — ONLINE',
];

export default function BootSequence() {
  const { displayed, done } = useTypewriter(BOOT_LINES, 25, 300);

  const getLineClass = (line) => {
    if (line.includes('CRITICAL')) return styles.lineDanger;
    if (line.includes('⚠') || line.includes('ANOMALIES')) return styles.lineWarning;
    if (line.includes('ONLINE')) return styles.lineFinal;
    return '';
  };

  return (
    <div className={styles.container}>
      <div className={styles.terminal}>
        {displayed.map((line, i) => (
          <div key={i} className={`${styles.line} ${getLineClass(line)}`} style={{ animationDelay: '0s' }}>
            {line}
          </div>
        ))}
        {!done && <span className={styles.cursor} />}
      </div>
    </div>
  );
}
