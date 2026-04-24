import styles from './BottomTimeline.module.css';

export default function BottomTimeline({ scenarioIndex, totalScenarios, goTo }) {
  return (
    <div className={styles.timeline}>
      {Array.from({ length: totalScenarios }, (_, i) => (
        <div
          key={i}
          className={`${styles.dot} ${i === scenarioIndex ? styles.dotActive : ''} ${i < scenarioIndex ? styles.dotVisited : ''}`}
          onClick={() => goTo(i)}
        />
      ))}
      <div className={styles.hint}>→ next · ← prev · F fullscreen</div>
    </div>
  );
}
