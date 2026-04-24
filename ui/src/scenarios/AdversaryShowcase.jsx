import { motion } from 'framer-motion';
import { GENERATIONS } from '../data/constants';
import styles from './AdversaryShowcase.module.css';

export default function AdversaryShowcase() {
  return (
    <div className={styles.container}>
      <div className={styles.left}>
        <div className={styles.title}>⚔️ HYDRA — 5 Sleeper Generations</div>
        {GENERATIONS.map((gen, i) => (
          <motion.div
            key={gen.gen}
            className={styles.card}
            initial={{ x: -60, opacity: 0 }}
            animate={{ x: 0, opacity: 1 }}
            transition={{ delay: i * 0.2, type: 'spring', stiffness: 120 }}
            style={{ borderColor: `${gen.color}30` }}
          >
            <div
              className={styles.genBadge}
              style={{ borderColor: gen.color, color: gen.color, background: `${gen.color}15` }}
            >
              {gen.gen}
            </div>
            <div className={styles.cardInfo}>
              <div className={styles.cardName}>{gen.icon} Gen-{gen.gen}: {gen.name}</div>
              <div className={styles.cardDesc}>{gen.desc}</div>
            </div>
            {gen.warning && <span className={styles.warningIcon}>⚠️</span>}
          </motion.div>
        ))}
      </div>

      <div className={styles.right}>
        <div className={styles.meterTitle}>Threat Level</div>
        <div className={styles.meterTrack}>
          <motion.div
            className={styles.meterFill}
            initial={{ height: '0%' }}
            animate={{ height: '100%' }}
            transition={{ duration: 2, delay: 0.5, ease: 'easeOut' }}
            style={{
              background: 'linear-gradient(to top, #22c55e, #eab308, #f97316, #ef4444, #dc2626)',
            }}
          />
        </div>
        <motion.div
          className={styles.meterLabel}
          style={{ color: 'var(--hydra-primary)' }}
          initial={{ opacity: 0 }}
          animate={{ opacity: 1 }}
          transition={{ delay: 2.5 }}
        >
          CRITICAL
        </motion.div>
      </div>
    </div>
  );
}
