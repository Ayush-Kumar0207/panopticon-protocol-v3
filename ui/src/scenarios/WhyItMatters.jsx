import { motion } from 'framer-motion';
import { CAPABILITIES } from '../data/constants';
import styles from './WhyItMatters.module.css';

export default function WhyItMatters() {
  return (
    <div className={styles.container}>
      <motion.div
        className={styles.title}
        initial={{ opacity: 0, y: -20 }}
        animate={{ opacity: 1, y: 0 }}
      >
        This trains the capabilities for{' '}
        <span className={styles.titleAccent}>Scalable AI Oversight</span>
      </motion.div>

      <div className={styles.grid}>
        {CAPABILITIES.map((cap, i) => (
          <motion.div
            key={cap.name}
            className={styles.card}
            initial={{ opacity: 0, y: 30 }}
            animate={{ opacity: 1, y: 0 }}
            transition={{ delay: i * 0.15, type: 'spring', stiffness: 120 }}
          >
            <div className={styles.cardIcon}>{cap.icon}</div>
            <div className={styles.cardContent}>
              <div className={styles.cardTitle}>{cap.name}</div>
              <div className={styles.cardDesc}>{cap.desc}</div>
            </div>
          </motion.div>
        ))}
      </div>

      <motion.div
        className={styles.footer}
        initial={{ opacity: 0 }}
        animate={{ opacity: 1 }}
        transition={{ delay: 0.8 }}
      >
        Theme: Multi-Agent Interactions → Fleet AI Scalable Oversight
      </motion.div>
    </div>
  );
}
