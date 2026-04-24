import { motion } from 'framer-motion';
import { TOOLS } from '../data/constants';
import styles from './ArsenalDisplay.module.css';

export default function ArsenalDisplay() {
  return (
    <div className={styles.container}>
      <motion.div
        className={styles.title}
        initial={{ opacity: 0, y: -20 }}
        animate={{ opacity: 1, y: 0 }}
      >
        🛡️ ARGUS Arsenal — 7 Counter-Intelligence Tools
      </motion.div>

      <div className={styles.grid}>
        {TOOLS.map((tool, i) => (
          <motion.div
            key={tool.name}
            className={`${styles.card} ${i === 6 ? styles.lastRow : ''}`}
            initial={{ scale: 0, opacity: 0 }}
            animate={{ scale: 1, opacity: 1 }}
            transition={{ delay: i * 0.12, type: 'spring', stiffness: 200 }}
          >
            <div className={styles.icon}>{tool.icon}</div>
            <div className={styles.name}>{tool.name}</div>
            <div className={styles.desc}>{tool.desc}</div>
          </motion.div>
        ))}
      </div>

      <motion.div
        className={styles.footer}
        initial={{ opacity: 0 }}
        animate={{ opacity: 1 }}
        transition={{ delay: 1.2 }}
      >
        + <span>HYDRA Adaptive Memory</span> — the adversary LEARNS your tactics
      </motion.div>
    </div>
  );
}
