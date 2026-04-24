import { useState, useEffect } from 'react';
import { AnimatePresence, motion } from 'framer-motion';
import { getHealth } from './api/client';

import CommandCenter from './tabs/CommandCenter';
import DemoMode from './tabs/DemoMode';
import TrainingEvidence from './tabs/TrainingEvidence';
import ArchitectureView from './tabs/ArchitectureView';

import styles from './App.module.css';

const TABS = [
  { id: 'command', label: '🎮 COMMAND CENTER', icon: '🎮' },
  { id: 'demo', label: '🤖 AI AGENT DEMO', icon: '🤖' },
  { id: 'training', label: '📈 TRAINING EVIDENCE', icon: '📈' },
  { id: 'architecture', label: '🏗️ ARCHITECTURE', icon: '🏗️' },
];

const TAB_COMPONENTS = {
  command: CommandCenter,
  demo: DemoMode,
  training: TrainingEvidence,
  architecture: ArchitectureView,
};

function App() {
  const [activeTab, setActiveTab] = useState('command');
  const [serverOnline, setServerOnline] = useState(false);

  useEffect(() => {
    let mounted = true;
    const check = async () => {
      try {
        await getHealth();
        if (mounted) setServerOnline(true);
      } catch {
        if (mounted) setServerOnline(false);
      }
    };
    check();
    const interval = setInterval(check, 10000);
    return () => { mounted = false; clearInterval(interval); };
  }, []);

  const TabComponent = TAB_COMPONENTS[activeTab];

  return (
    <div className={styles.app}>
      <div className={styles.bgGrid} />
      <div className={styles.scanline} />

      {/* ── Top Bar ── */}
      <div className={styles.topbar}>
        <div className={styles.brand}>
          <div className={`${styles.brandDot} ${!serverOnline ? styles.brandDotOffline : ''}`} />
          ARGUS OS v3.0
        </div>

        <div className={styles.topCenter}>
          {TABS.map(tab => (
            <button
              key={tab.id}
              className={`${styles.tabBtn} ${activeTab === tab.id ? styles.tabBtnActive : ''}`}
              onClick={() => setActiveTab(tab.id)}
            >
              {tab.label}
            </button>
          ))}
        </div>

        <div className={styles.statusBadge}>
          <div className={styles.statusDot} style={{ background: serverOnline ? 'var(--success)' : 'var(--danger)' }} />
          {serverOnline ? 'BACKEND ONLINE' : 'BACKEND OFFLINE'}
        </div>
      </div>

      {/* ── Main Content ── */}
      <div className={styles.main}>
        <AnimatePresence mode="wait">
          <motion.div
            key={activeTab}
            className={styles.tabContent}
            initial={{ opacity: 0, y: 10 }}
            animate={{ opacity: 1, y: 0 }}
            exit={{ opacity: 0, y: -10 }}
            transition={{ duration: 0.25 }}
          >
            <TabComponent serverOnline={serverOnline} />
          </motion.div>
        </AnimatePresence>
      </div>
    </div>
  );
}

export default App;
