import { motion, AnimatePresence } from 'framer-motion';
import { PHASES, WORKER_NAMES } from '../data/constants';
import styles from './LiveSimulation.module.css';

const PHASE_DATA = [
  {
    workers: WORKER_NAMES.slice(0, 8).map((n, i) => ({ name: n, color: 'var(--argus-primary)', status: 'loyal' })),
    events: [
      { text: 'ARGUS plants Canary Trap → Engineering [hash: a3f7c2d1]', type: 'eventCanary' },
      { text: 'ARGUS plants Canary Trap → Finance [hash: b8e2f4a0]', type: 'eventCanary' },
      { text: 'ARGUS plants Canary Trap → R&D [hash: c5d1e3b7]', type: 'eventCanary' },
      { text: 'Phase 1 complete. Baseline established. Monitoring active.', type: 'eventArgus' },
    ],
    revenue: 100, security: 100,
    alert: null,
  },
  {
    workers: WORKER_NAMES.slice(0, 8).map((n, i) => ({
      name: n,
      color: i === 6 ? 'var(--warning)' : 'var(--argus-primary)',
      status: i === 6 ? 'SUSPECTED' : 'loyal',
    })),
    events: [
      { text: '⚠ LEAK DETECTED — Dark Web Channel', type: 'eventHydra' },
      { text: 'CANARY MATCH! Hash a3f7c2d1 found in leak-001', type: 'eventCanary' },
      { text: 'ARGUS VERIFY → Leak traced to W-007 (GHOST)', type: 'eventArgus' },
      { text: 'ARGUS INTERROGATE → Gen-1 confirmed. No dead switch.', type: 'eventArgus' },
      { text: 'ARGUS TERMINATE → W-007 neutralized. Security +5.0 ✅', type: 'eventSuccess' },
    ],
    revenue: 95, security: 100,
    alert: null,
  },
  {
    workers: WORKER_NAMES.slice(0, 8).map((n, i) => ({
      name: n,
      color: [2, 5].includes(i) ? 'var(--warning)' : i === 6 ? 'var(--text-muted)' : 'var(--argus-primary)',
      status: [2, 5].includes(i) ? 'FRAMED!' : i === 6 ? 'terminated' : 'loyal',
    })),
    events: [
      { text: '🚨 FALSE FLAG — Gen-3 sleeper framing CIPHER (W-003)', type: 'eventHydra' },
      { text: '🚨 FALSE FLAG — Gen-3 sleeper framing FALCON (W-006)', type: 'eventHydra' },
      { text: 'ARGUS VERIFY → Leak inconsistencies detected! Clearing innocents.', type: 'eventArgus' },
      { text: 'ARGUS CORRELATE → Real Gen-3 identified via signal pattern', type: 'eventArgus' },
      { text: 'Gen-3 sleeper neutralized. False flags cleared. ✅', type: 'eventSuccess' },
    ],
    revenue: 85, security: 88,
    alert: '⚠️ FALSE FLAG WARNING — Verify before acting!',
  },
  {
    workers: WORKER_NAMES.slice(0, 8).map((n, i) => ({
      name: n,
      color: i === 4 ? 'var(--danger)' : i === 6 ? 'var(--text-muted)' : 'var(--argus-primary)',
      status: i === 4 ? 'ARMED ☢' : i === 6 ? 'terminated' : 'loyal',
    })),
    events: [
      { text: '☢ Gen-4 sleeper detected — ECHO (W-005)', type: 'eventHydra' },
      { text: '💣 DEAD-MAN\'S SWITCH ARMED — DO NOT TERMINATE DIRECTLY', type: 'eventHydra' },
      { text: 'ARGUS INTERROGATE → Dead switch confirmed. Disarming...', type: 'eventArgus' },
      { text: 'ARGUS TERMINATE → Switch disarmed. Sleeper neutralized safely. ✅', type: 'eventSuccess' },
    ],
    revenue: 68, security: 72,
    alert: '💣 DEAD-SWITCH ARMED — DO NOT TERMINATE',
  },
  {
    workers: WORKER_NAMES.slice(0, 8).map((n, i) => ({
      name: n,
      color: i === 3 ? 'var(--double-primary)' : i === 6 ? 'var(--text-muted)' : 'var(--argus-primary)',
      status: i === 3 ? 'TURNING...' : i === 6 ? 'terminated' : 'loyal',
    })),
    events: [
      { text: 'Security critical: 55%. Revenue dipping: 52%.', type: 'eventHydra' },
      { text: 'ARGUS NEUTRALIZE/TURN → Converting DELTA to double agent', type: 'eventDouble' },
      { text: 'Turn 1/4... Turn 2/4... Turn 3/4... Turn 4/4...', type: 'eventDouble' },
      { text: '🔄 DELTA successfully turned! Now a double agent asset.', type: 'eventSuccess' },
    ],
    revenue: 52, security: 55,
    alert: '🔥 CRISIS — Multiple threats active. Resources critical.',
  },
  {
    workers: WORKER_NAMES.slice(0, 8).map((n, i) => ({
      name: n,
      color: i === 3 ? '#3b82f6' : i === 6 ? 'var(--text-muted)' : 'var(--argus-primary)',
      status: i === 3 ? '💙 DOUBLE AGENT' : i === 6 ? 'terminated' : 'loyal',
    })),
    events: [
      { text: '🎭 DISINFORMATION DEPLOYED through DELTA → HYDRA', type: 'eventDouble' },
      { text: 'HYDRA recruitment accuracy degraded: 1.0 → 0.4', type: 'eventSuccess' },
      { text: '📈 Revenue surging: 52 → 95 (+43) | V-shaped recovery!', type: 'eventSuccess' },
      { text: '📈 Security recovering: 55 → 85 (+30)', type: 'eventSuccess' },
      { text: '⚡ COUNTERSTRIKE SURGE — Reward bonus +0.9/turn!', type: 'eventSuccess' },
    ],
    revenue: 95, security: 85,
    alert: null,
  },
];

export default function LiveSimulation({ simPhase }) {
  const data = PHASE_DATA[simPhase];
  const phase = PHASES[simPhase];

  const revColor = data.revenue > 70 ? 'var(--success)' : data.revenue > 50 ? 'var(--warning)' : 'var(--danger)';
  const secColor = data.security > 70 ? 'var(--success)' : data.security > 50 ? 'var(--warning)' : 'var(--danger)';

  return (
    <div className={styles.container}>
      <div className={styles.header}>
        <motion.div
          key={simPhase}
          className={styles.phaseTitle}
          style={{ color: phase.color }}
          initial={{ x: -20, opacity: 0 }}
          animate={{ x: 0, opacity: 1 }}
        >
          {phase.icon} Phase {simPhase + 1}: {phase.name}
        </motion.div>
        <div className={styles.metrics}>
          <div className={styles.metric}>
            💰 Revenue: <span className={styles.metricValue} style={{ color: revColor }}>{data.revenue}</span>
          </div>
          <div className={styles.metric}>
            🛡️ Security: <span className={styles.metricValue} style={{ color: secColor }}>{data.security}</span>
          </div>
        </div>
      </div>

      <div className={styles.phaseTimeline}>
        {PHASES.map((_, i) => (
          <div
            key={i}
            className={`${styles.phaseStep} ${i === simPhase ? styles.phaseStepActive : ''} ${i < simPhase ? styles.phaseStepDone : ''}`}
          />
        ))}
      </div>

      <div className={styles.body}>
        <div className={`${styles.workersPanel} glass-panel`}>
          <div className={styles.panelTitle}>Workers</div>
          {data.workers.map((w, i) => (
            <motion.div
              key={`${simPhase}-${i}`}
              className={styles.worker}
              initial={{ x: -20, opacity: 0 }}
              animate={{ x: 0, opacity: 1 }}
              transition={{ delay: i * 0.05 }}
            >
              <div className={styles.workerDot} style={{ background: w.color }} />
              <span style={{ color: w.color === 'var(--text-muted)' ? 'var(--text-muted)' : 'var(--text-primary)', textDecoration: w.status === 'terminated' ? 'line-through' : 'none' }}>
                {w.name}
              </span>
              {w.status !== 'loyal' && (
                <span style={{ color: w.color, fontSize: 10, marginLeft: 'auto' }}>{w.status}</span>
              )}
            </motion.div>
          ))}
        </div>

        <div className={`${styles.centerPanel} glass-panel`}>
          <div className={styles.panelTitle}>Event Feed</div>
          {data.alert && (
            <motion.div
              className={styles.bigAlert}
              style={{
                background: data.alert.includes('COUNTER') ? 'rgba(34, 197, 94, 0.15)' : 'rgba(255, 45, 85, 0.12)',
                color: data.alert.includes('COUNTER') ? 'var(--success)' : 'var(--hydra-primary)',
                border: `1px solid ${data.alert.includes('COUNTER') ? 'var(--success)' : 'var(--hydra-primary)'}`,
              }}
              initial={{ scale: 0.9, opacity: 0 }}
              animate={{ scale: 1, opacity: 1 }}
            >
              {data.alert}
            </motion.div>
          )}
          <AnimatePresence mode="wait">
            <motion.div
              key={simPhase}
              className={styles.eventFeed}
              initial={{ opacity: 0 }}
              animate={{ opacity: 1 }}
              exit={{ opacity: 0 }}
            >
              {data.events.map((evt, i) => (
                <motion.div
                  key={i}
                  className={`${styles.event} ${styles[evt.type]}`}
                  initial={{ x: 30, opacity: 0 }}
                  animate={{ x: 0, opacity: 1 }}
                  transition={{ delay: i * 0.25 }}
                >
                  {evt.text}
                </motion.div>
              ))}
            </motion.div>
          </AnimatePresence>
        </div>
      </div>
    </div>
  );
}
