import { motion } from 'framer-motion';
import { AreaChart, Area, XAxis, YAxis, CartesianGrid, Tooltip, ResponsiveContainer } from 'recharts';
import trainingData from '../data/trainingData.json';
import styles from './TrainingResults.module.css';

const LEVEL_COLORS = {
  easy: '#22c55e',
  medium: '#3b82f6',
  hard: '#f59e0b',
  level_4: '#ef4444',
  level_5: '#a855f7',
};

const LEVEL_LABELS = {
  easy: 'Easy',
  medium: 'Medium',
  hard: 'Hard',
  level_4: 'Level 4',
  level_5: 'Level 5',
};

function buildChartData() {
  const points = trainingData.easy.skill.length;
  const data = [];
  for (let i = 0; i < points; i++) {
    const point = { skill: Math.round(trainingData.easy.skill[i] * 100) };
    for (const level of Object.keys(LEVEL_COLORS)) {
      point[level] = parseFloat(trainingData[level].reward_mean[i].toFixed(2));
    }
    data.push(point);
  }
  return data;
}

const chartData = buildChartData();

export default function TrainingResults() {
  const finalEasy = trainingData.easy.reward_mean[9].toFixed(1);
  const finalL5 = trainingData.level_5.reward_mean[9].toFixed(1);
  const caughtL5 = trainingData.level_5.caught_mean[9].toFixed(1);

  return (
    <div className={styles.container}>
      <motion.div
        className={styles.title}
        initial={{ opacity: 0, y: -20 }}
        animate={{ opacity: 1, y: 0 }}
      >
        📈 PPO Curriculum Training — Reward Curves Across 5 Levels
      </motion.div>

      <div className={styles.legend}>
        {Object.entries(LEVEL_LABELS).map(([key, label]) => (
          <div key={key} className={styles.legendItem}>
            <div className={styles.legendDot} style={{ background: LEVEL_COLORS[key] }} />
            {label}
          </div>
        ))}
      </div>

      <motion.div
        className={styles.chartContainer}
        initial={{ opacity: 0, scale: 0.95 }}
        animate={{ opacity: 1, scale: 1 }}
        transition={{ delay: 0.3 }}
      >
        <ResponsiveContainer width="100%" height="100%">
          <AreaChart data={chartData} margin={{ top: 10, right: 20, left: 0, bottom: 10 }}>
            <defs>
              {Object.entries(LEVEL_COLORS).map(([key, color]) => (
                <linearGradient key={key} id={`grad-${key}`} x1="0" y1="0" x2="0" y2="1">
                  <stop offset="5%" stopColor={color} stopOpacity={0.3} />
                  <stop offset="95%" stopColor={color} stopOpacity={0} />
                </linearGradient>
              ))}
            </defs>
            <CartesianGrid strokeDasharray="3 3" stroke="rgba(148,163,184,0.08)" />
            <XAxis
              dataKey="skill"
              tick={{ fill: '#94a3b8', fontSize: 12 }}
              label={{ value: 'Training Progress (%)', position: 'insideBottom', offset: -5, fill: '#94a3b8', fontSize: 12 }}
            />
            <YAxis
              tick={{ fill: '#94a3b8', fontSize: 12 }}
              label={{ value: 'Episode Reward', angle: -90, position: 'insideLeft', fill: '#94a3b8', fontSize: 12 }}
            />
            <Tooltip
              contentStyle={{ background: '#111827', border: '1px solid rgba(148,163,184,0.2)', borderRadius: 8, color: '#f1f5f9', fontFamily: 'JetBrains Mono', fontSize: 12 }}
            />
            {Object.entries(LEVEL_COLORS).map(([key, color]) => (
              <Area
                key={key}
                type="monotone"
                dataKey={key}
                stroke={color}
                strokeWidth={2}
                fill={`url(#grad-${key})`}
                animationDuration={1500}
                animationBegin={300}
              />
            ))}
          </AreaChart>
        </ResponsiveContainer>
      </motion.div>

      <div className={styles.statsRow}>
        <motion.div className={styles.statCard} initial={{ y: 20, opacity: 0 }} animate={{ y: 0, opacity: 1 }} transition={{ delay: 0.5 }}>
          <div className={styles.statLabel}>Security After Training</div>
          <div className={styles.statValue} style={{ color: 'var(--success)' }}>0% → 100%</div>
          <div className={styles.statSub}>Easy/Medium/Hard</div>
        </motion.div>
        <motion.div className={styles.statCard} initial={{ y: 20, opacity: 0 }} animate={{ y: 0, opacity: 1 }} transition={{ delay: 0.7 }}>
          <div className={styles.statLabel}>Sleepers Caught (L5)</div>
          <div className={styles.statValue} style={{ color: 'var(--double-primary)' }}>{caughtL5}/5</div>
          <div className={styles.statSub}>Manchurian difficulty</div>
        </motion.div>
        <motion.div className={styles.statCard} initial={{ y: 20, opacity: 0 }} animate={{ y: 0, opacity: 1 }} transition={{ delay: 0.9 }}>
          <div className={styles.statLabel}>Peak Reward (L5)</div>
          <div className={styles.statValue} style={{ color: 'var(--argus-primary)' }}>{finalL5}</div>
          <div className={styles.statSub}>With Counterstrike surge</div>
        </motion.div>
      </div>
    </div>
  );
}
