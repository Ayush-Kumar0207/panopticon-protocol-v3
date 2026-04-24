import { motion } from 'framer-motion';
import { QRCodeSVG } from 'qrcode.react';
import { HF_SPACE_URL } from '../data/constants';
import styles from './ArchitectureView.module.css';

const LAYERS = [
  {
    icon: '⚙️', title: 'Layer 1: Engine', color: 'var(--argus-primary)',
    desc: 'Pure Python game engine with 9-step turn cycle. Pydantic v2 state machine enforcing information asymmetry. Zero framework dependencies.',
    files: ['environment.py', 'models.py'],
  },
  {
    icon: '🌐', title: 'Layer 2: API', color: 'var(--success)',
    desc: 'FastAPI REST server with 11 endpoints. OpenEnv-compliant /reset, /step, /tasks, /grade/{task_id}. CORS enabled. Static dashboard hosting.',
    files: ['_server.py', 'tasks.py', 'grader.py'],
  },
  {
    icon: '🧠', title: 'Layer 3: Training', color: 'var(--canary-primary)',
    desc: 'Dual training paths: 3-head PPO with curriculum learning (native RL) + Qwen 2.5 fine-tuning via HuggingFace TRL/SFT with LoRA (LLM agent).',
    files: ['train_rl.py', 'train_trl.py', 'gym_wrapper.py'],
  },
  {
    icon: '📊', title: 'Layer 4: Evaluation', color: 'var(--double-primary)',
    desc: '5-dimension programmatic grader: Security (30%), Revenue (25%), Intelligence (20%), Adaptability (15%), Efficiency (10%). Dynamic weights per difficulty tier.',
    files: ['grader.py', 'smoke_test.py', 'e2e_verify.py'],
  },
];

const INNOVATIONS = [
  { icon: '🪤', title: 'Canary Traps', desc: 'Cryptographic hash tracking to trace information leaks back to specific workers' },
  { icon: '🧬', title: '5-Gen Sleepers', desc: 'Enemies evolve from amateur leakers to undetectable Manchurian candidates' },
  { icon: '🚩', title: 'False Flags', desc: 'Gen-3+ plants decoy signals that frame innocent workers' },
  { icon: '💣', title: 'Dead Switches', desc: 'Gen-4+ self-destruct on careless termination, requiring interrogation first' },
  { icon: '🔄', title: 'Double Agents', desc: '4-turn conversion process turns caught spies into intelligence assets' },
  { icon: '🎭', title: 'Disinformation', desc: 'Turned agents feed false intel back to HYDRA, degrading adversary accuracy' },
  { icon: '🧠', title: 'HYDRA Memory', desc: 'Adversary adapts counter-tactics based on agent behavior patterns' },
  { icon: '📐', title: '5D Grading', desc: 'Multi-dimensional rubric prevents reward gaming via single-metric exploitation' },
  { icon: '🎯', title: 'Partial Observability', desc: 'Agent sees public info only — must infer hidden states via investigation' },
];

const TECH = ['PyTorch', 'Gymnasium', 'Pydantic v2', 'FastAPI', 'HF TRL', 'PEFT/LoRA', 'React', 'Framer Motion', 'Recharts', 'Docker'];

export default function ArchitectureView() {
  return (
    <div className={styles.container}>
      <div className={`${styles.header} glass-panel`}>
        <div className={styles.title}>🏗️ System Architecture — 4-Layer Decoupled Design</div>
        <div className={styles.subtitle}>Each layer can be modified independently. Engine knows nothing about training. Training knows nothing about grading.</div>
      </div>

      <div className={styles.sectionTitle}>Architecture Layers</div>
      <div className={styles.layers}>
        {LAYERS.map((layer, i) => (
          <div key={i}>
            <motion.div className={`${styles.layer} glass-panel`}
              initial={{ x: -30, opacity: 0 }} animate={{ x: 0, opacity: 1 }} transition={{ delay: i * 0.12 }}
            >
              <div className={styles.layerIcon}>{layer.icon}</div>
              <div className={styles.layerContent}>
                <div className={styles.layerTitle} style={{ color: layer.color }}>{layer.title}</div>
                <div className={styles.layerDesc}>{layer.desc}</div>
                <div className={styles.layerFiles}>
                  {layer.files.map(f => <span key={f} className={styles.fileBadge}>{f}</span>)}
                </div>
              </div>
            </motion.div>
            {i < LAYERS.length - 1 && <div className={styles.arrow}>↕</div>}
          </div>
        ))}
      </div>

      <div className={styles.sectionTitle}>7 Innovation Mechanics + 2 Meta-Innovations</div>
      <div className={styles.innovations}>
        {INNOVATIONS.map((inn, i) => (
          <motion.div key={inn.title} className={`${styles.innovation} glass-panel`}
            initial={{ scale: 0.8, opacity: 0 }} animate={{ scale: 1, opacity: 1 }} transition={{ delay: 0.4 + i * 0.06 }}
          >
            <div className={styles.innovIcon}>{inn.icon}</div>
            <div className={styles.innovTitle}>{inn.title}</div>
            <div className={styles.innovDesc}>{inn.desc}</div>
          </motion.div>
        ))}
      </div>

      <div className={styles.sectionTitle}>Tech Stack</div>
      <div className={styles.techStack}>
        {TECH.map(t => <span key={t} className={styles.techBadge}>{t}</span>)}
      </div>

      <div className={`${styles.qrSection} glass-panel`}>
        <div className={styles.qrWrapper}>
          <QRCodeSVG value={HF_SPACE_URL} size={120} bgColor="#fff" fgColor="#0a0e1a" level="M" />
        </div>
        <div className={styles.qrInfo}>
          <div className={styles.qrTitle}>Try it Live → HuggingFace Space</div>
          <div className={styles.qrUrl}>{HF_SPACE_URL}</div>
        </div>
      </div>

      <div className={styles.tagline}>
        <span className={styles.taglineGradient}>Can your AI catch the spy?</span>
      </div>
    </div>
  );
}
