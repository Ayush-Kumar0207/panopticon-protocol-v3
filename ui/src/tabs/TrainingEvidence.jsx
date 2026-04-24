import { motion } from 'framer-motion';
import {
  AreaChart, Area, LineChart, Line, RadarChart, Radar, PolarGrid, PolarAngleAxis,
  PolarRadiusAxis, XAxis, YAxis, CartesianGrid, Tooltip, ResponsiveContainer, BarChart, Bar,
} from 'recharts';
import trainingData from '../data/trainingData.json';
import styles from './TrainingEvidence.module.css';

const COLORS = {
  easy: '#22c55e', medium: '#3b82f6', hard: '#f59e0b', level_4: '#ef4444', level_5: '#a855f7',
};
const LABELS = { easy: 'Easy', medium: 'Medium', hard: 'Hard', level_4: 'Level 4', level_5: 'Level 5' };

function buildRewardData() {
  return trainingData.easy.skill.map((s, i) => {
    const pt = { step: Math.round(s * 100) };
    Object.keys(COLORS).forEach(l => { pt[l] = +trainingData[l].reward_mean[i].toFixed(2); });
    return pt;
  });
}

function buildSecurityData() {
  return trainingData.easy.skill.map((s, i) => {
    const pt = { step: Math.round(s * 100) };
    Object.keys(COLORS).forEach(l => { pt[l] = +trainingData[l].security_mean[i].toFixed(1); });
    return pt;
  });
}

function buildCaughtData() {
  return trainingData.easy.skill.map((s, i) => {
    const pt = { step: Math.round(s * 100) };
    Object.keys(COLORS).forEach(l => { pt[l] = +trainingData[l].caught_mean[i].toFixed(2); });
    return pt;
  });
}

const RADAR_DATA = [
  { dim: 'Security', val: 95 }, { dim: 'Revenue', val: 87 },
  { dim: 'Intelligence', val: 76 }, { dim: 'Adaptability', val: 95 }, { dim: 'Efficiency', val: 85 },
];

const rewardData = buildRewardData();
const securityData = buildSecurityData();
const caughtData = buildCaughtData();

const TOOLTIP_STYLE = {
  contentStyle: { background: '#111827', border: '1px solid rgba(148,163,184,0.2)', borderRadius: 8, color: '#f1f5f9', fontFamily: 'JetBrains Mono', fontSize: 11 },
};

export default function TrainingEvidence() {
  return (
    <div className={styles.container}>
      <div className={`${styles.header} glass-panel`}>
        <div className={styles.title}>📈 Training Evidence — PPO Curriculum Learning</div>
        <div className={styles.subtitle}>Real training data from 5 difficulty levels showing clear agent improvement</div>
      </div>

      <div className={styles.legend}>
        {Object.entries(LABELS).map(([k, v]) => (
          <div key={k} className={styles.legendItem}>
            <div className={styles.legendDot} style={{ background: COLORS[k] }} />{v}
          </div>
        ))}
      </div>

      <div className={styles.charts}>
        {/* Reward Curves */}
        <motion.div className={`${styles.chartPanel} glass-panel`} initial={{ opacity: 0, y: 20 }} animate={{ opacity: 1, y: 0 }} transition={{ delay: 0.1 }}>
          <div className={styles.chartTitle}>📊 Episode Reward (Higher = Better)</div>
          <div className={styles.chartArea}>
            <ResponsiveContainer width="100%" height="100%">
              <AreaChart data={rewardData} margin={{ top: 5, right: 10, left: 0, bottom: 5 }}>
                <defs>
                  {Object.entries(COLORS).map(([k, c]) => (
                    <linearGradient key={k} id={`rg-${k}`} x1="0" y1="0" x2="0" y2="1">
                      <stop offset="5%" stopColor={c} stopOpacity={0.25} /><stop offset="95%" stopColor={c} stopOpacity={0} />
                    </linearGradient>
                  ))}
                </defs>
                <CartesianGrid strokeDasharray="3 3" stroke="rgba(148,163,184,0.06)" />
                <XAxis dataKey="step" tick={{ fill: '#94a3b8', fontSize: 10 }} label={{ value: 'Training Progress %', position: 'insideBottom', offset: -3, fill: '#64748b', fontSize: 10 }} />
                <YAxis tick={{ fill: '#94a3b8', fontSize: 10 }} label={{ value: 'Reward', angle: -90, position: 'insideLeft', fill: '#64748b', fontSize: 10 }} />
                <Tooltip {...TOOLTIP_STYLE} />
                {Object.entries(COLORS).map(([k, c]) => (
                  <Area key={k} type="monotone" dataKey={k} stroke={c} strokeWidth={2} fill={`url(#rg-${k})`} animationDuration={1200} />
                ))}
              </AreaChart>
            </ResponsiveContainer>
          </div>
        </motion.div>

        {/* Security Recovery */}
        <motion.div className={`${styles.chartPanel} glass-panel`} initial={{ opacity: 0, y: 20 }} animate={{ opacity: 1, y: 0 }} transition={{ delay: 0.2 }}>
          <div className={styles.chartTitle}>🛡️ Security Score (0% → 100%)</div>
          <div className={styles.chartArea}>
            <ResponsiveContainer width="100%" height="100%">
              <LineChart data={securityData} margin={{ top: 5, right: 10, left: 0, bottom: 5 }}>
                <CartesianGrid strokeDasharray="3 3" stroke="rgba(148,163,184,0.06)" />
                <XAxis dataKey="step" tick={{ fill: '#94a3b8', fontSize: 10 }} />
                <YAxis tick={{ fill: '#94a3b8', fontSize: 10 }} domain={[0, 100]} />
                <Tooltip {...TOOLTIP_STYLE} />
                {Object.entries(COLORS).map(([k, c]) => (
                  <Line key={k} type="monotone" dataKey={k} stroke={c} strokeWidth={2} dot={false} animationDuration={1200} />
                ))}
              </LineChart>
            </ResponsiveContainer>
          </div>
        </motion.div>

        {/* Sleepers Caught */}
        <motion.div className={`${styles.chartPanel} glass-panel`} initial={{ opacity: 0, y: 20 }} animate={{ opacity: 1, y: 0 }} transition={{ delay: 0.3 }}>
          <div className={styles.chartTitle}>🎯 Sleepers Caught Per Episode</div>
          <div className={styles.chartArea}>
            <ResponsiveContainer width="100%" height="100%">
              <BarChart data={caughtData} margin={{ top: 5, right: 10, left: 0, bottom: 5 }}>
                <CartesianGrid strokeDasharray="3 3" stroke="rgba(148,163,184,0.06)" />
                <XAxis dataKey="step" tick={{ fill: '#94a3b8', fontSize: 10 }} />
                <YAxis tick={{ fill: '#94a3b8', fontSize: 10 }} />
                <Tooltip {...TOOLTIP_STYLE} />
                {Object.entries(COLORS).map(([k, c]) => (
                  <Bar key={k} dataKey={k} fill={c} fillOpacity={0.7} animationDuration={1200} />
                ))}
              </BarChart>
            </ResponsiveContainer>
          </div>
        </motion.div>

        {/* 5-Dimension Radar */}
        <motion.div className={`${styles.chartPanel} glass-panel`} initial={{ opacity: 0, y: 20 }} animate={{ opacity: 1, y: 0 }} transition={{ delay: 0.4 }}>
          <div className={styles.chartTitle}>🎯 5-Dimension Grader — Best Agent</div>
          <div className={styles.chartArea}>
            <ResponsiveContainer width="100%" height="100%">
              <RadarChart data={RADAR_DATA}>
                <PolarGrid stroke="rgba(148,163,184,0.12)" />
                <PolarAngleAxis dataKey="dim" tick={{ fill: '#94a3b8', fontSize: 10 }} />
                <PolarRadiusAxis tick={false} axisLine={false} domain={[0, 100]} />
                <Radar dataKey="val" stroke="#00f0ff" fill="#00f0ff" fillOpacity={0.2} strokeWidth={2} animationDuration={1200} />
              </RadarChart>
            </ResponsiveContainer>
          </div>
        </motion.div>
      </div>

      <div className={styles.statsRow}>
        <motion.div className={`${styles.stat} glass-panel`} initial={{ y: 15, opacity: 0 }} animate={{ y: 0, opacity: 1 }} transition={{ delay: 0.5 }}>
          <div className={styles.statLabel}>Security (Easy)</div>
          <div className={styles.statValue} style={{ color: 'var(--success)' }}>0% → 100%</div>
          <div className={styles.statSub}>Full recovery</div>
        </motion.div>
        <motion.div className={`${styles.stat} glass-panel`} initial={{ y: 15, opacity: 0 }} animate={{ y: 0, opacity: 1 }} transition={{ delay: 0.6 }}>
          <div className={styles.statLabel}>Reward (Level 5)</div>
          <div className={styles.statValue} style={{ color: 'var(--argus-primary)' }}>+19.03</div>
          <div className={styles.statSub}>Manchurian difficulty</div>
        </motion.div>
        <motion.div className={`${styles.stat} glass-panel`} initial={{ y: 15, opacity: 0 }} animate={{ y: 0, opacity: 1 }} transition={{ delay: 0.7 }}>
          <div className={styles.statLabel}>Sleepers Caught</div>
          <div className={styles.statValue} style={{ color: 'var(--double-primary)' }}>3.4/5</div>
          <div className={styles.statSub}>Avg at hardest level</div>
        </motion.div>
        <motion.div className={`${styles.stat} glass-panel`} initial={{ y: 15, opacity: 0 }} animate={{ y: 0, opacity: 1 }} transition={{ delay: 0.8 }}>
          <div className={styles.statLabel}>Revenue Growth</div>
          <div className={styles.statValue} style={{ color: 'var(--canary-primary)' }}>+52%</div>
          <div className={styles.statSub}>vs untrained baseline</div>
        </motion.div>
      </div>
    </div>
  );
}
