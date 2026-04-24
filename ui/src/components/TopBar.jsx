import styles from './TopBar.module.css';

const SCENARIO_NAMES = [
  'BOOT', 'OVERVIEW', 'ADVERSARY', 'ARSENAL',
  'LIVE SIM', 'TRAINING', 'OVERSIGHT', 'FINALE'
];

export default function TopBar({ scenarioIndex }) {
  return (
    <div className={styles.topbar}>
      <div className={styles.brand}>
        <div className={styles.dot} />
        ARGUS OS v3.0
      </div>
      <div className={styles.scenarioLabel}>
        SCENARIO <span>{scenarioIndex + 1}/8</span> — {SCENARIO_NAMES[scenarioIndex]}
      </div>
    </div>
  );
}
