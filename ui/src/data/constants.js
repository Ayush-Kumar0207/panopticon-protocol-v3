export const DEPARTMENTS = ['Engineering', 'Finance', 'R&D', 'Operations', 'Executive', 'Legal'];

export const WORKER_NAMES = [
  'ATLAS', 'BEACON', 'CIPHER', 'DELTA', 'ECHO',
  'FALCON', 'GHOST', 'HAWK', 'IRON', 'JADE'
];

export const LEAK_CHANNELS = ['Dark Web', 'Market Chatter', 'Competitor Filing', 'Press Leak', 'Insider Trade'];

export const PHASES = [
  { name: 'Orientation', icon: '🛡️', color: 'var(--argus-primary)' },
  { name: 'First Contact', icon: '📡', color: 'var(--canary-primary)' },
  { name: 'Escalation', icon: '⚠️', color: 'var(--warning)' },
  { name: 'Deep Cover', icon: '💣', color: 'var(--danger)' },
  { name: 'Crisis', icon: '🔥', color: 'var(--hydra-primary)' },
  { name: 'Counterstrike', icon: '⚡', color: 'var(--success)' },
];

export const GENERATIONS = [
  { gen: 1, name: 'Amateur', color: '#22c55e', icon: '🟢', desc: 'Basic data leaker — always trips canary traps', danger: 20 },
  { gen: 2, name: 'Professional', color: '#eab308', icon: '🟡', desc: 'Avoids canary departments & monitored channels', danger: 40 },
  { gen: 3, name: 'Spy Network', color: '#f97316', icon: '🟠', desc: 'Plants FALSE FLAGS to frame innocent workers', danger: 60, warning: true },
  { gen: 4, name: 'Terror Cell', color: '#ef4444', icon: '🔴', desc: 'DEAD-MAN\'S SWITCH — explodes on careless termination', danger: 80, warning: true },
  { gen: 5, name: 'Manchurian', color: '#dc2626', icon: '💀', desc: 'Top performer. Perfect loyalty. Secretly a spy.', danger: 100 },
];

export const TOOLS = [
  { name: 'Canary Trap', icon: '🪤', desc: 'Plant traceable data to reveal leakers', color: 'var(--canary-primary)' },
  { name: 'Monitor', icon: '📡', desc: 'Scan 5 leak channels for canary hash matches', color: 'var(--argus-primary)' },
  { name: 'Investigate', icon: '🔍', desc: 'Audit workers, verify leaks, correlate signals', color: 'var(--argus-primary)' },
  { name: 'Interrogate', icon: '💬', desc: 'Reveal generation level & dead-switch status', color: 'var(--warning)' },
  { name: 'Neutralize', icon: '⚡', desc: 'Terminate confirmed threats from the network', color: 'var(--danger)' },
  { name: 'Turn Agent', icon: '🔄', desc: 'Convert caught spies into double agents (4 turns)', color: 'var(--double-primary)' },
  { name: 'Disinformation', icon: '🎭', desc: 'Feed false intel back to HYDRA through double agents', color: 'var(--double-primary)' },
];

export const CAPABILITIES = [
  { name: 'Theory of Mind', icon: '🧠', desc: 'Modeling hidden worker states from partial observations' },
  { name: 'Deception Detection', icon: '🎭', desc: 'Separating real leaks from Gen-3 false flags' },
  { name: 'Strategic Planning', icon: '🗺️', desc: 'Multi-step canary → monitor → verify → terminate chains' },
  { name: 'Adaptive Response', icon: '🔄', desc: 'Re-calibrating strategy as HYDRA\'s memory evolves' },
];

export const HF_SPACE_URL = 'https://huggingface.co/spaces/Ayush-Kumar0207/panopticon-protocol-v3';
