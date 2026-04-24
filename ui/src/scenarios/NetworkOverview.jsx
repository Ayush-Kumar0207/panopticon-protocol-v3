import { motion } from 'framer-motion';
import { useState, useEffect } from 'react';
import { DEPARTMENTS } from '../data/constants';
import styles from './NetworkOverview.module.css';

const NODE_POSITIONS = [
  { x: 310, y: 60 },
  { x: 540, y: 160 },
  { x: 500, y: 350 },
  { x: 190, y: 350 },
  { x: 130, y: 160 },
  { x: 310, y: 240 },
];

const ICONS = ['⚙️', '💰', '🔬', '🏭', '👔', '⚖️'];
const INFECTED = [1, 3, 5];

const CONNECTIONS = [
  [0, 1], [0, 4], [1, 2], [2, 3], [3, 4], [0, 5], [1, 5], [2, 5], [3, 5], [4, 5],
];

export default function NetworkOverview() {
  const [showInfection, setShowInfection] = useState(false);

  useEffect(() => {
    const timer = setTimeout(() => setShowInfection(true), 2000);
    return () => clearTimeout(timer);
  }, []);

  return (
    <div className={styles.container}>
      <div className={styles.graph}>
        <svg className={styles.connections} viewBox="0 0 700 500">
          {CONNECTIONS.map(([a, b], i) => (
            <motion.line
              key={i}
              x1={NODE_POSITIONS[a].x + 32}
              y1={NODE_POSITIONS[a].y + 32}
              x2={NODE_POSITIONS[b].x + 32}
              y2={NODE_POSITIONS[b].y + 32}
              stroke="rgba(0, 240, 255, 0.15)"
              strokeWidth="1"
              initial={{ pathLength: 0 }}
              animate={{ pathLength: 1 }}
              transition={{ duration: 1, delay: i * 0.1 }}
            />
          ))}
        </svg>

        {DEPARTMENTS.map((dept, i) => {
          const infected = showInfection && INFECTED.includes(i);
          return (
            <motion.div
              key={dept}
              className={styles.node}
              style={{ left: NODE_POSITIONS[i].x, top: NODE_POSITIONS[i].y }}
              initial={{ scale: 0, opacity: 0 }}
              animate={{ scale: 1, opacity: 1 }}
              transition={{ type: 'spring', delay: i * 0.15, stiffness: 200 }}
            >
              <div className={`${styles.circle} ${infected ? styles.circleInfected : styles.circleSafe}`}>
                {ICONS[i]}
              </div>
              <span className={styles.label}>{dept}</span>
            </motion.div>
          );
        })}
      </div>

      <motion.div
        className={styles.tagline}
        initial={{ opacity: 0, y: 20 }}
        animate={{ opacity: 1, y: 0 }}
        transition={{ delay: 2.5, duration: 0.8 }}
      >
        <span className={styles.taglineAccent}>Among Us</span>… for AIs
      </motion.div>
    </div>
  );
}
