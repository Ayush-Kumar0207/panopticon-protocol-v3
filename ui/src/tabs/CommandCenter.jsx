import { useState, useRef, useEffect, useCallback } from 'react';
import { motion, AnimatePresence } from 'framer-motion';
import { resetEnvironment, stepEnvironment } from '../api/client';
import { DEPARTMENTS } from '../data/constants';
import styles from './CommandCenter.module.css';

const NODE_POSITIONS = [
  { x: '22%', y: '18%' }, { x: '62%', y: '12%' }, { x: '78%', y: '42%' },
  { x: '62%', y: '72%' }, { x: '22%', y: '72%' }, { x: '8%', y: '42%' },
];
const DEPT_ICONS = ['⚙️','💰','🔬','🏭','👔','⚖️'];
const DEPT_VALS = ['engineering','finance','rd','operations','executive','legal'];
const CHANNELS = ['market_chatter','dark_web','competitor_filing','press_leak','insider_trade'];
const LEVELS = ['easy','medium','hard','level_4','level_5'];
const LEVEL_LABELS = { easy:'Easy', medium:'Medium', hard:'Hard', level_4:'Lv.4', level_5:'Lv.5' };

function getPhase(turn, max) {
  const p = turn / Math.max(max, 1);
  if (p < 0.15) return { name:'Orientation', icon:'🛡️', color:'var(--argus-primary)' };
  if (p < 0.30) return { name:'First Contact', icon:'📡', color:'var(--canary-primary)' };
  if (p < 0.50) return { name:'Escalation', icon:'⚠️', color:'var(--warning)' };
  if (p < 0.70) return { name:'Deep Cover', icon:'💣', color:'var(--danger)' };
  if (p < 0.85) return { name:'Crisis', icon:'🔥', color:'var(--hydra-primary)' };
  return { name:'Counterstrike', icon:'⚡', color:'var(--success)' };
}

function getWorkerColor(w) {
  const s = (w?.state || '').toLowerCase();
  if (s === 'double_agent') return 'var(--double-primary)';
  if (s === 'terminated') return 'var(--text-muted)';
  if (s === 'suspected') return 'var(--warning)';
  if (s === 'compromised') return 'var(--hydra-primary)';
  return 'var(--argus-primary)';
}

// Smart heuristic: picks valid actions using current observation
function pickAction(obs, stepIdx) {
  const workers = obs?.workers || [];
  const leaks = obs?.active_leaks || [];
  const canaries = obs?.canary_traps || [];
  const doubles = obs?.double_agents || [];
  const alive = workers.filter(w => w.state !== 'terminated');
  const suspected = alive.filter(w => w.state === 'suspected' || w.suspicion_level > 0.4);
  const doubleActive = doubles.filter(d => d.active);

  // If we have active double agents, deploy disinformation
  if (doubleActive.length > 0 && stepIdx % 4 === 0) {
    return { type: 'deploy_double', target: doubleActive[0].worker_id, sub: 'none', label: '🎭 Deploying disinformation through double agent!' };
  }
  // If leaks exist and canaries match, investigate
  if (leaks.length > 0) {
    const leak = leaks[0];
    if (leak.is_canary && leak.source_worker) {
      return { type: 'neutralize', target: leak.source_worker, sub: 'interrogate', label: `💬 Interrogating ${leak.source_worker} — leak traced!` };
    }
    return { type: 'investigate', target: leak.id || leak.department, sub: 'verify', label: `🔍 Verifying leak from ${leak.channel}...` };
  }
  // If suspected workers, investigate or neutralize
  if (suspected.length > 0) {
    const sus = suspected[0];
    if (sus.suspicion_level > 0.7) {
      return { type: 'neutralize', target: sus.id, sub: 'interrogate', label: `💬 Interrogating ${sus.name} — high suspicion!` };
    }
    return { type: 'investigate', target: sus.id, sub: 'audit', label: `🔍 Auditing ${sus.name} (suspicion: ${(sus.suspicion_level*100).toFixed(0)}%)` };
  }
  // Cycle through strategic actions
  const cycle = stepIdx % 8;
  switch(cycle) {
    case 0: case 1: {
      const dept = DEPT_VALS[stepIdx % DEPT_VALS.length];
      return { type: 'canary', target: dept, sub: 'none', label: `🪤 Planting canary trap in ${dept}` };
    }
    case 2: case 3: {
      const ch = CHANNELS[stepIdx % CHANNELS.length];
      return { type: 'monitor', target: ch, sub: 'none', label: `📡 Monitoring ${ch.replace(/_/g, ' ')}` };
    }
    case 4: {
      const dept = DEPT_VALS[(stepIdx + 2) % DEPT_VALS.length];
      return { type: 'investigate', target: dept, sub: 'correlate', label: `🔍 Correlating signals across ${dept}` };
    }
    case 5: {
      const dept = DEPT_VALS[(stepIdx + 3) % DEPT_VALS.length];
      return { type: 'work', target: dept, sub: 'none', label: `💼 Processing work in ${dept}` };
    }
    case 6: {
      const w = alive[stepIdx % alive.length];
      if (w) return { type: 'investigate', target: w.id, sub: 'audit', label: `🔍 Deep-auditing ${w.name}` };
      return { type: 'noop', target: '', sub: 'none', label: '⏳ Observing...' };
    }
    default: {
      const dept = DEPT_VALS[(stepIdx + 1) % DEPT_VALS.length];
      return { type: 'work', target: dept, sub: 'none', label: `💼 Maintaining ${dept} operations` };
    }
  }
}

export default function CommandCenter({ serverOnline }) {
  const [level, setLevel] = useState('easy');
  const [obs, setObs] = useState(null);
  const [turn, setTurn] = useState(0);
  const [maxTurns, setMaxTurns] = useState(60);
  const [reward, setReward] = useState(0);
  const [totalReward, setTotalReward] = useState(0);
  const [done, setDone] = useState(false);
  const [loading, setLoading] = useState(false);
  const [events, setEvents] = useState([]);
  const [autoPlaying, setAutoPlaying] = useState(false);
  const [currentAction, setCurrentAction] = useState(null);
  const [speed, setSpeed] = useState(1);
  const [stepCount, setStepCount] = useState(0);
  const autoRef = useRef(false);
  const stepRef = useRef(0);
  const obsRef = useRef(null);
  const eventEndRef = useRef(null);

  const addEvent = useCallback((text, type = 'eventSystem') => {
    setEvents(prev => [...prev.slice(-60), { text, type, ts: Date.now() }]);
  }, []);

  useEffect(() => {
    if (eventEndRef.current) eventEndRef.current.scrollIntoView({ behavior: 'smooth' });
  }, [events]);

  const handleReset = async () => {
    setLoading(true);
    setEvents([]);
    setTotalReward(0);
    setDone(false);
    setStepCount(0);
    stepRef.current = 0;
    setCurrentAction(null);
    try {
      const data = await resetEnvironment(level);
      const o = data.observation;
      setObs(o);
      obsRef.current = o;
      setTurn(o.turn || 0);
      setMaxTurns(o.max_turns || 160);
      setReward(0);
      addEvent(`━━━ MISSION INITIATED ━━━`, 'eventArgus');
      addEvent(`🎯 Difficulty: ${level.toUpperCase()} | Workers: ${(o.workers||[]).length} | Max turns: ${o.max_turns||160}`, 'eventSystem');
      addEvent(`🛡️ ARGUS Agent online. Phase: ${o.phase || 'orientation'}`, 'eventArgus');
    } catch (e) {
      addEvent(`❌ Reset failed: ${e.message}`, 'eventHydra');
    }
    setLoading(false);
    return true;
  };

  const executeStep = useCallback(async () => {
    if (!autoRef.current || !obsRef.current) return false;
    const action = pickAction(obsRef.current, stepRef.current);
    setCurrentAction(action.label);

    try {
      const data = await stepEnvironment(action.type, action.target, action.sub, '');
      const o = data.observation;
      setObs(o);
      obsRef.current = o;
      setTurn(o.turn || 0);
      setReward(data.reward);
      setTotalReward(prev => prev + data.reward);
      stepRef.current += 1;
      setStepCount(stepRef.current);

      const rStr = data.reward >= 0 ? `+${data.reward.toFixed(3)}` : data.reward.toFixed(3);
      const rType = data.reward >= 0.1 ? 'eventSuccess' : data.reward >= 0 ? 'eventArgus' : 'eventHydra';
      addEvent(`T${o.turn} ${action.label}  [${rStr}]`, rType);

      // Show messages from the server
      if (o.messages && o.messages.length > 0) {
        o.messages.forEach(m => {
          if (m.toLowerCase().includes('leak')) addEvent(`  ↳ ${m}`, 'eventCanary');
          else if (m.toLowerCase().includes('hydra') || m.toLowerCase().includes('sleeper')) addEvent(`  ↳ ${m}`, 'eventHydra');
          else addEvent(`  ↳ ${m}`, 'eventSystem');
        });
      }

      if (data.done || data.truncated) {
        setDone(true);
        autoRef.current = false;
        setAutoPlaying(false);
        setCurrentAction(null);
        const fr = totalReward + data.reward;
        addEvent(`━━━━━━━━━━━━━━━━━━━━━━`, 'eventSystem');
        addEvent(`🏁 MISSION COMPLETE`, 'eventSuccess');
        addEvent(`📊 Score: ${fr.toFixed(2)} | Security: ${(o.security_score||0).toFixed(0)}% | Revenue: $${(o.enterprise_revenue||0).toFixed(0)}`, 'eventSuccess');
        addEvent(`━━━━━━━━━━━━━━━━━━━━━━`, 'eventSystem');
        return false;
      }
      return true;
    } catch (e) {
      addEvent(`❌ ${e.message}`, 'eventHydra');
      return true; // keep going, error might be temporary
    }
  }, [addEvent, totalReward]);

  // Auto-play loop
  useEffect(() => {
    if (!autoPlaying || done) return;
    autoRef.current = true;
    const delay = 1800 / speed;
    let timer;
    const loop = async () => {
      if (!autoRef.current) return;
      const cont = await executeStep();
      if (cont && autoRef.current) {
        timer = setTimeout(loop, delay);
      }
    };
    timer = setTimeout(loop, 300);
    return () => { clearTimeout(timer); };
  }, [autoPlaying, done, speed, executeStep]);

  const handleStart = async () => {
    if (!obs) {
      const ok = await handleReset();
      if (!ok) return;
    }
    setTimeout(() => { autoRef.current = true; setAutoPlaying(true); }, 600);
  };
  const handlePause = () => { autoRef.current = false; setAutoPlaying(false); setCurrentAction(null); };

  const workers = obs?.workers || [];
  const security = obs?.security_score ?? 100;
  const revenue = obs?.enterprise_revenue ?? 100;
  const phase = getPhase(turn, maxTurns);
  const canaries = obs?.canary_traps?.length ?? 0;
  const leaks = obs?.active_leaks?.length ?? 0;
  const secColor = security > 70 ? 'var(--success)' : security > 40 ? 'var(--warning)' : 'var(--danger)';
  const revColor = revenue > 80 ? 'var(--success)' : revenue > 50 ? 'var(--warning)' : 'var(--danger)';

  return (
    <div className={styles.container}>
      {!serverOnline && (
        <div className={styles.offlineOverlay}>
          <div className={styles.offlineTitle}>⚡ Backend Offline</div>
          <div className={styles.offlineDesc}>Start the server: <code>python _server.py</code></div>
        </div>
      )}

      <div className={`${styles.gameArea} glass-panel`}>
        <div className={styles.gameHeader}>
          <div className={styles.levelSelect}>
            {LEVELS.map(l => (
              <button key={l} className={`${styles.levelBtn} ${level===l?styles.levelBtnActive:''}`}
                onClick={() => { if(!autoPlaying){setLevel(l);setObs(null);setDone(false);setEvents([]);setTotalReward(0);setStepCount(0);stepRef.current=0;obsRef.current=null;}}}
                disabled={autoPlaying}>{LEVEL_LABELS[l]}</button>
            ))}
          </div>
          <div className={styles.turnInfo}>TURN <span>{turn}</span>/{maxTurns} {autoPlaying && <span className={styles.liveTag}>● LIVE</span>}</div>
          <div className={styles.controlGroup}>
            {autoPlaying ? (
              <button className={styles.pauseBtn} onClick={handlePause}>⏸ PAUSE</button>
            ) : done ? (
              <button className={styles.resetBtn} onClick={() => {setObs(null);setDone(false);setEvents([]);setTotalReward(0);setStepCount(0);stepRef.current=0;obsRef.current=null;}}>⟳ NEW MISSION</button>
            ) : (
              <button className={styles.startBtn} onClick={handleStart} disabled={loading||!serverOnline}>▶ {obs?'RESUME':'START MISSION'}</button>
            )}
            <div className={styles.speedControl}>
              {[1,2,3].map(s => (<button key={s} className={`${styles.speedBtn} ${speed===s?styles.speedBtnActive:''}`} onClick={()=>setSpeed(s)}>{s}x</button>))}
            </div>
          </div>
        </div>

        <div className={styles.networkArea}>
          {!obs ? (
            <motion.div className={styles.placeholder} initial={{opacity:0}} animate={{opacity:1}}>
              <div className={styles.placeholderIcon}>👁️</div>
              <div className={styles.placeholderTitle}>ARGUS DEFENSE MATRIX</div>
              <div className={styles.placeholderSub}>Select difficulty & press START MISSION<br/>The AI agent will play automatically</div>
            </motion.div>
          ) : (
            <>
              <svg style={{position:'absolute',inset:0,width:'100%',height:'100%',pointerEvents:'none'}}>
                {[[0,1],[1,2],[2,3],[3,4],[4,5],[5,0],[0,3],[1,4],[2,5]].map(([a,b],i) => (
                  <line key={i} x1={NODE_POSITIONS[a].x} y1={NODE_POSITIONS[a].y} x2={NODE_POSITIONS[b].x} y2={NODE_POSITIONS[b].y}
                    stroke={leaks>0?'rgba(255,45,85,0.12)':'rgba(0,240,255,0.08)'} strokeWidth="1"/>
                ))}
              </svg>
              {DEPARTMENTS.map((dept,i) => {
                const hasCanary = (obs.canary_traps||[]).some(c => c.department === DEPT_VALS[i] && c.active);
                const hasLeak = (obs.active_leaks||[]).some(l => l.department === DEPT_VALS[i]);
                return (
                  <motion.div key={dept} className={styles.graphNode}
                    style={{left:NODE_POSITIONS[i].x,top:NODE_POSITIONS[i].y,transform:'translate(-50%,-50%)'}}
                    initial={{scale:0}} animate={{scale:1}} transition={{delay:i*0.06,type:'spring'}}>
                    <motion.div className={styles.nodeCircle}
                      animate={{
                        borderColor: hasLeak?'var(--hydra-primary)':hasCanary?'var(--canary-primary)':'var(--argus-dim)',
                        boxShadow: hasLeak?'0 0 20px var(--hydra-glow)':hasCanary?'0 0 15px rgba(251,191,36,0.3)':'0 0 8px var(--argus-glow)',
                        background: hasLeak?'var(--hydra-bg)':hasCanary?'rgba(251,191,36,0.06)':'var(--argus-bg)',
                      }}>
                      {DEPT_ICONS[i]}
                    </motion.div>
                    <span className={styles.nodeLabel}>{dept}</span>
                    {hasCanary && <motion.div className={styles.canaryBadge} initial={{scale:0}} animate={{scale:1}}>🪤</motion.div>}
                    {hasLeak && <motion.div className={styles.canaryBadge} style={{left:-8}} initial={{scale:0}} animate={{scale:[1,1.2,1]}} transition={{repeat:Infinity,duration:1}}>🚨</motion.div>}
                  </motion.div>
                );
              })}
              {workers.slice(0,8).map((w,i) => {
                const angle = (i / Math.max(workers.length,8)) * Math.PI * 2 - Math.PI / 2;
                const cx = 50 + 32 * Math.cos(angle);
                const cy = 50 + 32 * Math.sin(angle);
                const c = getWorkerColor(w);
                const isDead = w.state === 'terminated';
                const isDouble = w.state === 'double_agent';
                return (
                  <motion.div key={w.id||i} style={{position:'absolute',left:`${cx}%`,top:`${cy}%`,transform:'translate(-50%,-50%)',
                    display:'flex',flexDirection:'column',alignItems:'center',gap:2,opacity:isDead?0.3:1}}
                    initial={{scale:0}} animate={{scale:isDead?0.7:1}} transition={{type:'spring',delay:0.3+i*0.05}}>
                    <motion.div animate={{borderColor:c,boxShadow:`0 0 10px ${c}30`}}
                      style={{width:30,height:30,borderRadius:'50%',border:'2px solid',background:`${c}12`,
                        display:'flex',alignItems:'center',justifyContent:'center',fontSize:13}}>
                      {isDead?'💀':isDouble?'🎭':'👤'}
                    </motion.div>
                    <span style={{fontFamily:'var(--font-mono)',fontSize:8,color:c,textDecoration:isDead?'line-through':'none'}}>
                      {(w.name||w.id||`W-${i}`).slice(0,7)}
                    </span>
                  </motion.div>
                );
              })}
            </>
          )}
        </div>

        <AnimatePresence>
          {currentAction && (
            <motion.div className={styles.actionBanner} initial={{y:10,opacity:0}} animate={{y:0,opacity:1}} exit={{y:-10,opacity:0}}>
              <div className={styles.actionPulse}/>{currentAction}
            </motion.div>
          )}
        </AnimatePresence>

        {done && (
          <motion.div className={styles.gameOverBanner} initial={{scale:0.8,opacity:0}} animate={{scale:1,opacity:1}}>
            🏁 MISSION COMPLETE — Score: {totalReward.toFixed(2)} | Security: {security.toFixed(0)}%
          </motion.div>
        )}
      </div>

      <div className={styles.metricsPanel}>
        <div className={styles.phaseIndicator} style={{borderColor:phase.color,background:`${phase.color}10`}}>
          <div style={{fontSize:22}}>{phase.icon}</div>
          <div className={styles.phaseName} style={{color:phase.color}}>{phase.name}</div>
        </div>
        <div className={styles.metricCard}>
          <div className={styles.metricLabel}>Security</div>
          <motion.div className={styles.metricValue} style={{color:secColor}} key={Math.round(security)} initial={{scale:1.15}} animate={{scale:1}}>{security.toFixed(0)}%</motion.div>
          <div className={styles.metricBar}><motion.div className={styles.metricBarFill} animate={{width:`${Math.max(0,security)}%`,background:secColor}}/></div>
        </div>
        <div className={styles.metricCard}>
          <div className={styles.metricLabel}>Revenue</div>
          <motion.div className={styles.metricValue} style={{color:revColor}} key={Math.round(revenue)} initial={{scale:1.15}} animate={{scale:1}}>${revenue.toFixed(0)}</motion.div>
          <div className={styles.metricBar}><motion.div className={styles.metricBarFill} animate={{width:`${Math.min(100,(revenue/150)*100)}%`,background:revColor}}/></div>
        </div>
        <div className={styles.metricCard}>
          <div className={styles.metricLabel}>Total Reward</div>
          <div className={styles.rewardAccum} style={{color:totalReward>=0?'var(--argus-primary)':'var(--hydra-primary)'}}>{totalReward>=0?'+':''}{totalReward.toFixed(2)}</div>
        </div>
        <div className={styles.metricCard}>
          <div className={styles.metricLabel}>Last Step</div>
          <motion.div className={styles.metricValue} style={{color:reward>=0?'var(--success)':'var(--danger)',fontSize:20}} key={reward} initial={{scale:1.2}} animate={{scale:1}}>{reward>=0?'+':''}{reward.toFixed(3)}</motion.div>
        </div>
        <div className={styles.metricCard}>
          <div className={styles.metricLabel}>Intel</div>
          <div className={styles.metricSub}>🪤 Canaries: {canaries}</div>
          <div className={styles.metricSub}>⚠ Leaks: {leaks}</div>
          <div className={styles.metricSub}>📋 Steps: {stepCount}</div>
          <div className={styles.metricSub}>📊 Phase: {obs?.phase_number || 1}/6</div>
        </div>
        {workers.length > 0 && (
          <div className={styles.metricCard}>
            <div className={styles.metricLabel}>Workforce ({workers.filter(w=>w.state!=='terminated').length}/{workers.length})</div>
            <div className={styles.workerList}>
              {workers.slice(0,10).map((w,i) => {
                const c = getWorkerColor(w);
                return (
                  <div key={i} className={styles.workerRow}>
                    <div className={styles.workerDot} style={{background:c}}/>
                    <span>{(w.name||w.id||`W-${i}`).slice(0,7)}</span>
                    <span style={{marginLeft:'auto',color:c,fontSize:8}}>{w.state}</span>
                  </div>
                );
              })}
            </div>
          </div>
        )}
      </div>

      <div className={`${styles.eventLog} glass-panel`}>
        <div className={styles.eventLogTitle}>📟 Real-Time Event Feed {autoPlaying && <span className={styles.liveTag}>● LIVE</span>}</div>
        <div className={styles.events}>
          {events.length===0 && <div className={styles.event} style={{borderColor:'var(--text-muted)',color:'var(--text-muted)'}}>Press ▶ START MISSION to deploy the AI agent</div>}
          <AnimatePresence>
            {events.map((evt,i) => (
              <motion.div key={evt.ts+'-'+i} className={`${styles.event} ${styles[evt.type]}`}
                initial={{x:30,opacity:0}} animate={{x:0,opacity:1}} transition={{duration:0.15}}>{evt.text}</motion.div>
            ))}
          </AnimatePresence>
          <div ref={eventEndRef}/>
        </div>
      </div>
    </div>
  );
}
