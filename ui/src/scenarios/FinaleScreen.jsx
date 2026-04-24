import { motion } from 'framer-motion';
import { QRCodeSVG } from 'qrcode.react';
import { Radar, RadarChart, PolarGrid, PolarAngleAxis, PolarRadiusAxis, ResponsiveContainer } from 'recharts';
import { HF_SPACE_URL } from '../data/constants';
import styles from './FinaleScreen.module.css';

const RADAR_DATA = [
  { dimension: 'Security', value: 95.5, fullMark: 100 },
  { dimension: 'Revenue', value: 87.0, fullMark: 100 },
  { dimension: 'Intelligence', value: 76.0, fullMark: 100 },
  { dimension: 'Adaptability', value: 94.9, fullMark: 100 },
  { dimension: 'Efficiency', value: 85.0, fullMark: 100 },
];

export default function FinaleScreen() {
  return (
    <div className={styles.container}>
      <div className={styles.top}>
        <motion.div
          className={styles.radarSection}
          initial={{ opacity: 0, scale: 0.8 }}
          animate={{ opacity: 1, scale: 1 }}
          transition={{ delay: 0.2, type: 'spring' }}
        >
          <div className={styles.radarTitle}>5-Dimension Grader</div>
          <ResponsiveContainer width={280} height={250}>
            <RadarChart data={RADAR_DATA}>
              <PolarGrid stroke="rgba(148,163,184,0.15)" />
              <PolarAngleAxis dataKey="dimension" tick={{ fill: '#94a3b8', fontSize: 11 }} />
              <PolarRadiusAxis tick={false} axisLine={false} domain={[0, 100]} />
              <Radar
                name="Score"
                dataKey="value"
                stroke="#00f0ff"
                fill="#00f0ff"
                fillOpacity={0.2}
                strokeWidth={2}
                animationDuration={1500}
              />
            </RadarChart>
          </ResponsiveContainer>
        </motion.div>

        <motion.div
          className={styles.qrSection}
          initial={{ opacity: 0, scale: 0.5 }}
          animate={{ opacity: 1, scale: 1 }}
          transition={{ delay: 0.5, type: 'spring', stiffness: 150 }}
        >
          <div className={styles.qrWrapper}>
            <QRCodeSVG
              value={HF_SPACE_URL}
              size={180}
              bgColor="#ffffff"
              fgColor="#0a0e1a"
              level="M"
            />
          </div>
          <div className={styles.qrLabel}>
            Try it live → <span className={styles.qrLabelAccent}>HuggingFace Space</span>
          </div>
        </motion.div>

        <motion.div
          className={styles.teamSection}
          initial={{ opacity: 0, x: 30 }}
          animate={{ opacity: 1, x: 0 }}
          transition={{ delay: 0.4 }}
        >
          <div className={styles.teamTitle}>Team</div>
          <div className={styles.teamName}>Ayush Kumar</div>
          <div className={styles.teamName}>Ravi Prashant</div>
          <div className={styles.teamSub}>Meta PyTorch OpenEnv Hackathon</div>
        </motion.div>
      </div>

      <motion.div
        className={styles.stats}
        initial={{ opacity: 0 }}
        animate={{ opacity: 1 }}
        transition={{ delay: 0.8 }}
      >
        <div><span>7</span> Mechanics</div>
        <div><span>6</span> Phases</div>
        <div><span>5</span> Difficulty Levels</div>
        <div><span>2</span> Training Paths</div>
        <div><span>~3,800</span> Lines of Code</div>
      </motion.div>

      <motion.div
        className={styles.tagline}
        initial={{ opacity: 0, y: 30 }}
        animate={{ opacity: 1, y: 0 }}
        transition={{ delay: 1.0, type: 'spring' }}
      >
        <span className={styles.taglineAccent}>Can your AI catch the spy?</span>
      </motion.div>
    </div>
  );
}
