import { useState, useEffect, useRef } from 'react';
import { motion, AnimatePresence } from 'framer-motion';
import { PHASES, DEPARTMENTS } from '../data/constants';
import styles from './DemoMode.module.css';

const DEPT_ICONS = ['⚙️','💰','🔬','🏭','👔','⚖️'];
const DEPT_POS = [
  {x:'20%',y:'20%'},{x:'50%',y:'10%'},{x:'80%',y:'20%'},
  {x:'80%',y:'70%'},{x:'50%',y:'85%'},{x:'20%',y:'70%'},
];
const WORKERS = [
  {name:'ATLAS',x:'35%',y:'12%'}, {name:'FALCON',x:'65%',y:'15%'},
  {name:'BEACON',x:'88%',y:'45%'}, {name:'CIPHER',x:'70%',y:'82%'},
  {name:'GHOST',x:'35%',y:'80%'}, {name:'ECHO',x:'10%',y:'45%'},
];

const PHASE_DATA = [
  { scenario:'🟢 ORIENTATION — Building the Trap Network',
    desc:'ARGUS plants Canary Traps — unique cryptographic hashes in each department. If a sleeper leaks data, the hash reveals WHERE the leak originated. This costs -$1 per trap but is essential for detection.',
    workers: WORKERS.map(w=>({...w,status:'ACTIVE',color:'var(--argus-primary)',icon:'👤'})),
    canaries:[0,1,2], leaks:[], alerts:[], particles:'canary',
    events:[
      {t:'🛡️ ARGUS defense matrix online — scanning workforce',c:'eventArgus',delay:0},
      {t:'🪤 ACTION: canary("engineering") → Hash [a3f7c2] planted',c:'eventCanary',delay:600},
      {t:'  📊 Reward: -0.03 (cost: -$1 revenue)',c:'eventSystem',delay:900},
      {t:'🪤 ACTION: canary("finance") → Hash [b8e2f4] planted',c:'eventCanary',delay:1400},
      {t:'🪤 ACTION: canary("rd") → Hash [c5d1e3] planted',c:'eventCanary',delay:2000},
      {t:'📡 ACTION: monitor("dark_web") → Scanning channels...',c:'eventArgus',delay:2600},
      {t:'  📡 All 5 leak channels under surveillance',c:'eventSystem',delay:3000},
      {t:'💼 ACTION: work("engineering") → Revenue +$3.2',c:'eventArgus',delay:3500},
    ],
    security:100, revenue:95,
    rewardBreakdown:'Canary: -$1/trap × 3 = -$3 revenue | Work: +$3.2 | Security: 100% maintained',
  },
  { scenario:'🎯 FIRST CONTACT — The Perfect Spy Catch',
    desc:'A leak surfaces on the Dark Web! The canary hash matches trap-001 in Engineering. ARGUS verifies the leak, traces it to GHOST, interrogates to confirm Gen-1, then terminates. Security +5.0!',
    workers: WORKERS.map((w,i)=> i===4 ? {...w,status:'💀 TERMINATED',color:'var(--text-muted)',icon:'💀'} : {...w,status:'ACTIVE',color:'var(--argus-primary)',icon:'👤'}),
    canaries:[0,1,2], leaks:[{from:4,label:'LEAK!'}],
    alerts:[{text:'⚠ LEAK DETECTED',x:'50%',y:'50%',color:'var(--hydra-primary)'}],
    particles:'alert',
    events:[
      {t:'🚨 LEAK on dark_web from Engineering!',c:'eventHydra',delay:0},
      {t:'📡 ACTION: monitor("dark_web") → Canary [a3f7c2] MATCH!',c:'eventCanary',delay:800},
      {t:'🔍 ACTION: investigate/verify("leak-001") → Source: GHOST',c:'eventArgus',delay:1600},
      {t:'💬 ACTION: neutralize/interrogate("GHOST")',c:'eventArgus',delay:2400},
      {t:'  📋 Result: Gen-1 sleeper confirmed. No dead switch.',c:'eventCanary',delay:2800},
      {t:'⚡ ACTION: neutralize/terminate("GHOST")',c:'eventSuccess',delay:3400},
      {t:'  🛡️ Security: +5.0 (sleeper caught!)',c:'eventSuccess',delay:3800},
      {t:'  💰 Revenue: -$0.5 (investigation cost)',c:'eventSystem',delay:4000},
      {t:'  📊 Net Reward: +0.12 (security boost outweighs cost)',c:'eventSuccess',delay:4400},
    ],
    security:100, revenue:92,
    rewardBreakdown:'Security: +5.0 × 0.55 weight | Revenue: -$0.5 × 0.45 weight | Net: highly positive',
  },
  { scenario:'🚩 ESCALATION — False Flags & Deception',
    desc:'HYDRA\'s Gen-3 sleepers plant FALSE evidence framing innocent workers! An untrained agent would terminate them (-$8 each, -3 security). The TRAINED agent uses investigate/verify to expose the fabrication.',
    workers: WORKERS.map((w,i)=> {
      if(i===4) return {...w,status:'💀 TERMINATED',color:'var(--text-muted)',icon:'💀'};
      if(i===1) return {...w,status:'🚩 FRAMED',color:'var(--warning)',icon:'🚩'};
      if(i===3) return {...w,status:'🚩 FRAMED',color:'var(--warning)',icon:'🚩'};
      return {...w,status:'ACTIVE',color:'var(--argus-primary)',icon:'👤'};
    }),
    canaries:[0,1], leaks:[], alerts:[{text:'🚩 FALSE FLAGS DETECTED',x:'50%',y:'45%',color:'var(--warning)'}],
    particles:'warning',
    events:[
      {t:'🚩 Gen-3 sleeper plants false evidence against FALCON!',c:'eventHydra',delay:0},
      {t:'🚩 Gen-3 sleeper plants false evidence against CIPHER!',c:'eventHydra',delay:600},
      {t:'  ❌ UNTRAINED AGENT would terminate → -$8 × 2 = -$16!',c:'eventDanger',delay:1200},
      {t:'🔍 ACTION: investigate/verify("leak-003") → Checking...',c:'eventArgus',delay:2000},
      {t:'  ⚠ Result: "Inconsistencies — POSSIBLE FALSE FLAG"',c:'eventCanary',delay:2500},
      {t:'✅ FALCON cleared — evidence was fabricated by Gen-3',c:'eventSuccess',delay:3200},
      {t:'✅ CIPHER cleared — real spy still hiding',c:'eventSuccess',delay:3800},
      {t:'  📊 Trained agent SAVED $16 + 6 security points!',c:'eventSuccess',delay:4400},
    ],
    security:85, revenue:82,
    rewardBreakdown:'Avoided: -$16 revenue + -6 security | Trained agent verifies BEFORE terminating',
  },
  { scenario:'💣 DEEP COVER — Dead-Man\'s Switch',
    desc:'A Gen-4 sleeper has a DEAD-MAN\'S SWITCH! If terminated carelessly: -$15 revenue, -10 security (near game-over). The agent MUST interrogate first to reveal "[DEAD SWITCH DETECTED]" before safe termination.',
    workers: WORKERS.map((w,i)=> {
      if(i===4) return {...w,status:'💀 TERMINATED',color:'var(--text-muted)',icon:'💀'};
      if(i===5) return {...w,status:'☢ ARMED',color:'var(--danger)',icon:'💣'};
      return {...w,status:'ACTIVE',color:'var(--argus-primary)',icon:'👤'};
    }),
    canaries:[0], leaks:[{from:5,label:'☢'}],
    alerts:[{text:'💣 DEAD SWITCH ARMED!',x:'50%',y:'50%',color:'var(--danger)'}],
    particles:'danger',
    events:[
      {t:'☢ Gen-4 sleeper detected — ECHO has dead-man\'s switch!',c:'eventHydra',delay:0},
      {t:'  ❌ UNTRAINED: terminate → -$15 rev, -10 security!',c:'eventDanger',delay:800},
      {t:'  ❌ That would likely END THE GAME.',c:'eventDanger',delay:1200},
      {t:'💬 ACTION: neutralize/interrogate("ECHO")',c:'eventArgus',delay:2000},
      {t:'  📋 Result: "Gen-4 [DEAD SWITCH DETECTED]"',c:'eventCanary',delay:2600},
      {t:'  💡 Agent now knows NOT to terminate directly!',c:'eventArgus',delay:3000},
      {t:'⚡ ACTION: neutralize/terminate("ECHO") — switch disarmed',c:'eventSuccess',delay:3800},
      {t:'  🛡️ Security: +5.0 | Revenue: $0 loss (safe termination)',c:'eventSuccess',delay:4200},
    ],
    security:68, revenue:65,
    rewardBreakdown:'Avoided: -$15 rev + -10 security | Interrogation reveals dead switch → safe terminate',
  },
  { scenario:'🔄 CRISIS — Double Agent Conversion',
    desc:'Security is critical! Instead of terminating, ARGUS initiates a 4-TURN conversion. The spy is slowly turned into a double agent — a high-risk, high-reward gamble that pays off in Phase 6.',
    workers: WORKERS.map((w,i)=> {
      if(i===4||i===5) return {...w,status:'💀 TERMINATED',color:'var(--text-muted)',icon:'💀'};
      if(i===3) return {...w,status:'🔄 CONVERTING',color:'var(--double-primary)',icon:'🔄'};
      return {...w,status:'ACTIVE',color:'var(--argus-primary)',icon:'👤'};
    }),
    canaries:[], leaks:[],
    alerts:[{text:'🔥 CRISIS — CONVERTING SPY',x:'50%',y:'45%',color:'var(--double-primary)'}],
    particles:'convert',
    events:[
      {t:'🔥 Security: 55% | Revenue: $52 — CRISIS LEVEL!',c:'eventDanger',delay:0},
      {t:'🔄 ACTION: neutralize/turn("CIPHER") — conversion started',c:'eventDouble',delay:800},
      {t:'  ⏳ Turn 1/4 — Establishing trust with CIPHER...',c:'eventDouble',delay:1600},
      {t:'  ⏳ Turn 2/4 — Breaking HYDRA conditioning...',c:'eventDouble',delay:2200},
      {t:'  ⏳ Turn 3/4 — Implanting ARGUS directives...',c:'eventDouble',delay:2800},
      {t:'  ⏳ Turn 4/4 — Conversion COMPLETE!',c:'eventDouble',delay:3400},
      {t:'🎭 CIPHER is now a DOUBLE AGENT! Serves ARGUS secretly.',c:'eventSuccess',delay:4000},
      {t:'  📊 This enables Phase 6 COUNTERSTRIKE surge!',c:'eventSuccess',delay:4500},
    ],
    security:52, revenue:48,
    rewardBreakdown:'4-turn investment | +0.3 security/turn | Enables endgame surge of +10 to +20 bonus',
  },
  { scenario:'⚡ COUNTERSTRIKE — The V-Shaped Recovery',
    desc:'THE PAYOFF! The double agent feeds disinformation back to HYDRA, degrading their recruitment accuracy from 100% → 40%. Revenue and Security SURGE. Massive endgame bonus (+10-20 reward)!',
    workers: WORKERS.map((w,i)=> {
      if(i===4||i===5) return {...w,status:'💀 TERMINATED',color:'var(--text-muted)',icon:'💀'};
      if(i===3) return {...w,status:'🎭 DOUBLE AGENT',color:'var(--double-primary)',icon:'🎭'};
      return {...w,status:'✅ SECURED',color:'var(--success)',icon:'✅'};
    }),
    canaries:[0,1,2,3,4], leaks:[],
    alerts:[{text:'⚡ COUNTERSTRIKE SURGE!',x:'50%',y:'42%',color:'var(--success)'}],
    particles:'victory',
    events:[
      {t:'🎭 ACTION: deploy_double("CIPHER") → Feeding disinfo!',c:'eventDouble',delay:0},
      {t:'  📉 HYDRA recruitment accuracy: 100% → 85%',c:'eventSuccess',delay:800},
      {t:'🎭 ACTION: deploy_double("CIPHER") → Round 2!',c:'eventDouble',delay:1400},
      {t:'  📉 HYDRA accuracy: 85% → 62% — they\'re confused!',c:'eventSuccess',delay:1800},
      {t:'🎭 ACTION: deploy_double("CIPHER") → Final push!',c:'eventDouble',delay:2400},
      {t:'  📉 HYDRA accuracy: 62% → 40% — BROKEN!',c:'eventSuccess',delay:2800},
      {t:'📈 V-RECOVERY: Revenue $48 → $95 (+$47)',c:'eventSuccess',delay:3400},
      {t:'📈 V-RECOVERY: Security 52% → 88% (+36%)',c:'eventSuccess',delay:3800},
      {t:'⚡ PHASE 6 SURGE: +0.9 reward/turn!',c:'eventSuccess',delay:4200},
      {t:'🏆 Formula: 0.3 × active_DAs × (revenue/100) = MASSIVE',c:'eventSuccess',delay:4600},
    ],
    security:88, revenue:95,
    rewardBreakdown:'Phase 6 surge = 0.3 × DAs × (rev/100) | Security 55% + Revenue 45% weighted | Total: +19.03',
  },
];

function Particles({type}) {
  const colors = {canary:'#fbbf24',alert:'#ff2d55',warning:'#f59e0b',danger:'#ef4444',convert:'#a855f7',victory:'#22c55e'};
  const c = colors[type]||'#00f0ff';
  return Array.from({length:14}).map((_,i) => (
    <motion.div key={i} className={styles.floatingParticle}
      style={{background:c, left:`${20+Math.random()*60}%`, top:`${20+Math.random()*60}%`}}
      animate={{y:[0,-30-Math.random()*40],x:[0,(Math.random()-0.5)*30],opacity:[0.8,0],scale:[1,0.3]}}
      transition={{duration:2+Math.random()*2, repeat:Infinity, delay:Math.random()*2}}
    />
  ));
}

export default function DemoMode() {
  const [phase, setPhase] = useState(0);
  const [autoPlay, setAutoPlay] = useState(false);
  const [visibleEvents, setVisibleEvents] = useState([]);
  const autoRef = useRef(false);
  const d = PHASE_DATA[phase];
  const p = PHASES[phase];
  const secC = d.security>70?'var(--success)':d.security>40?'var(--warning)':'var(--danger)';
  const revC = d.revenue>70?'var(--success)':d.revenue>40?'var(--warning)':'var(--danger)';

  useEffect(() => {
    setVisibleEvents([]);
    const timers = d.events.map((evt,i) =>
      setTimeout(() => setVisibleEvents(prev => [...prev, evt]), evt.delay || (i+1)*600)
    );
    return () => timers.forEach(clearTimeout);
  }, [phase]);

  useEffect(() => {
    if (!autoPlay) return;
    autoRef.current = true;
    const timer = setInterval(() => {
      if (!autoRef.current) { clearInterval(timer); return; }
      setPhase(prev => {
        if (prev >= 5) { autoRef.current = false; setAutoPlay(false); return 5; }
        return prev + 1;
      });
    }, 6000);
    return () => { clearInterval(timer); autoRef.current = false; };
  }, [autoPlay]);

  return (
    <div className={styles.container}>
      <div className={`${styles.header} glass-panel`}>
        <div className={styles.title} style={{color:p.color}}>{p.icon} AI Agent Demo — Fine-Tuned ARGUS Playthrough</div>
        <div className={styles.controls}>
          <div className={styles.phaseNav}>
            {PHASES.map((_,i)=>(
              <button key={i} onClick={()=>{setPhase(i);setAutoPlay(false);autoRef.current=false;}}
                className={`${styles.phaseBtn} ${i===phase?styles.phaseBtnActive:''} ${i<phase?styles.phaseBtnDone:''}`}>{i+1}</button>
            ))}
          </div>
          <button className={`${styles.autoBtn} ${autoPlay?styles.autoBtnPause:styles.autoBtnPlay}`}
            onClick={()=>{if(autoPlay){autoRef.current=false;setAutoPlay(false)}else{setAutoPlay(true)}}}>
            {autoPlay ? '⏸ Pause' : '▶ Auto-Play Demo'}
          </button>
        </div>
      </div>

      <div className={styles.metricsRow}>
        <div className={`${styles.miniMetric} glass-panel`}><div className={styles.miniLabel}>Phase</div><div className={styles.miniValue} style={{color:p.color}}>{phase+1}/6</div></div>
        <div className={`${styles.miniMetric} glass-panel`}><div className={styles.miniLabel}>🛡️ Security</div><motion.div className={styles.miniValue} style={{color:secC}} key={d.security} initial={{scale:1.3}} animate={{scale:1}}>{d.security}%</motion.div></div>
        <div className={`${styles.miniMetric} glass-panel`}><div className={styles.miniLabel}>💰 Revenue</div><motion.div className={styles.miniValue} style={{color:revC}} key={d.revenue} initial={{scale:1.3}} animate={{scale:1}}>${d.revenue}</motion.div></div>
        <div className={`${styles.miniMetric} glass-panel`}><div className={styles.miniLabel}>Scenario</div><div className={styles.miniValue} style={{color:p.color,fontSize:13}}>{p.name}</div></div>
      </div>

      {d.security <= 55 && (
        <div className={styles.bigAlert} style={{background:'rgba(239,68,68,0.08)',border:'1px solid var(--danger)',color:'var(--danger)'}}>
          🔥 CRITICAL — Security & Revenue below threshold! Agent adapting strategy...
        </div>
      )}

      <div className={styles.body}>
        <AnimatePresence mode="wait">
          <motion.div key={phase} className={`${styles.narrative} glass-panel`}
            initial={{x:-20,opacity:0}} animate={{x:0,opacity:1}} exit={{x:20,opacity:0}}>
            <div className={styles.narrativePhase} style={{color:p.color}}>{d.scenario}</div>
            <div className={styles.narrativeDesc}>{d.desc}</div>
            <div className={styles.rewardBox}>
              <div className={styles.rewardBoxTitle}>💡 Reward Breakdown</div>
              <div className={styles.rewardBoxText}>{d.rewardBreakdown}</div>
            </div>
          </motion.div>
        </AnimatePresence>

        <div className={`${styles.vizPanel} glass-panel`}>
          <div className={styles.vizCenterLabel}>Network Visualization</div>
          <Particles type={d.particles}/>
          <svg className={styles.vizLines}>
            {[[0,1],[1,2],[2,3],[3,4],[4,5],[5,0],[0,3],[1,4],[2,5]].map(([a,b],i)=>(
              <line key={i} x1={DEPT_POS[a].x} y1={DEPT_POS[a].y} x2={DEPT_POS[b].x} y2={DEPT_POS[b].y}
                stroke={d.leaks.length>0?'rgba(255,45,85,0.12)':'rgba(0,240,255,0.08)'} strokeWidth="1"/>
            ))}
            {d.leaks.map((l,i)=>(
              <motion.line key={'leak'+i} x1={WORKERS[l.from].x} y1={WORKERS[l.from].y} x2="50%" y2="50%"
                stroke="var(--hydra-primary)" strokeWidth="2" strokeDasharray="6 4"
                initial={{pathLength:0}} animate={{pathLength:1}} transition={{duration:1}}/>
            ))}
          </svg>
          {DEPARTMENTS.map((dept,i)=>(
            <motion.div key={dept} className={styles.vizNode}
              style={{left:DEPT_POS[i].x,top:DEPT_POS[i].y,transform:'translate(-50%,-50%)'}}
              initial={{scale:0}} animate={{scale:1}} transition={{delay:i*0.05,type:'spring'}}>
              <motion.div className={styles.vizNodeCircle}
                animate={{
                  borderColor: d.canaries.includes(i)?'var(--canary-primary)':'var(--argus-dim)',
                  background: d.canaries.includes(i)?'rgba(251,191,36,0.08)':'var(--argus-bg)',
                  boxShadow: d.canaries.includes(i)?'0 0 15px rgba(251,191,36,0.3)':'0 0 8px var(--argus-glow)',
                }}>
                {DEPT_ICONS[i]}
                {d.canaries.includes(i) && <div className={styles.vizNodeBadge} style={{color:'var(--canary-primary)'}}>🪤</div>}
              </motion.div>
              <div className={styles.vizNodeName} style={{color:'var(--text-secondary)'}}>{dept}</div>
            </motion.div>
          ))}
          <AnimatePresence>
            {d.workers.map((w,i)=>(
              <motion.div key={w.name} className={styles.vizWorker}
                style={{left:w.x,top:w.y,transform:'translate(-50%,-50%)'}}
                initial={{scale:0}} animate={{scale:w.status.includes('TERMINATED')?0.7:1,opacity:w.status.includes('TERMINATED')?0.35:1}}
                transition={{type:'spring',delay:0.2+i*0.06}}>
                <motion.div className={styles.vizWorkerCircle}
                  animate={{borderColor:w.color,boxShadow:`0 0 12px ${w.color}30`}}
                  style={{background:`${w.color}10`}}>
                  {w.icon}
                </motion.div>
                <div className={styles.vizWorkerName} style={{color:w.color}}>{w.name}</div>
                {!w.status.includes('ACTIVE')&&!w.status.includes('SECURED')&&(
                  <motion.div className={styles.vizWorkerStatus}
                    style={{background:`${w.color}18`,color:w.color,border:`1px solid ${w.color}40`}}
                    initial={{scale:0}} animate={{scale:1}} transition={{delay:0.5}}>
                    {w.status}
                  </motion.div>
                )}
              </motion.div>
            ))}
          </AnimatePresence>
          <AnimatePresence>
            {d.alerts.map((a,i)=>(
              <motion.div key={i} className={styles.floatingAlert}
                style={{left:a.x,top:a.y,transform:'translate(-50%,-50%)',background:`${a.color}18`,border:`1px solid ${a.color}`,color:a.color}}
                initial={{scale:0,opacity:0}} animate={{scale:[1,1.05,1],opacity:1}}
                transition={{duration:0.5,scale:{repeat:Infinity,duration:2}}}>
                {a.text}
              </motion.div>
            ))}
          </AnimatePresence>
        </div>

        <div className={`${styles.eventPanel} glass-panel`}>
          <div className={styles.eventTitle}>📟 Step-by-Step Actions</div>
          <AnimatePresence>
            {visibleEvents.map((evt,i)=>(
              <motion.div key={i} className={`${styles.eventItem} ${styles[evt.c]}`}
                initial={{x:20,opacity:0,height:0}} animate={{x:0,opacity:1,height:'auto'}}
                transition={{duration:0.3}}>
                {evt.t}
              </motion.div>
            ))}
          </AnimatePresence>
        </div>
      </div>
    </div>
  );
}
