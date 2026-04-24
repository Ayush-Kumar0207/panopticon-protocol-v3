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
  { desc:'ARGUS boots up and deploys canary traps — cryptographic data packets planted into departments. When information leaks, the unique hash reveals exactly who leaked it.',
    workers: WORKERS.map(w=>({...w,status:'ACTIVE',color:'var(--argus-primary)',icon:'👤'})),
    canaries:[0,1,2], leaks:[], alerts:[], particles:'canary',
    events:[
      {t:'🛡️ ARGUS defense matrix online',c:'eventArgus'},
      {t:'🪤 Canary [a3f7c2] → Engineering',c:'eventCanary'},
      {t:'🪤 Canary [b8e2f4] → Finance',c:'eventCanary'},
      {t:'🪤 Canary [c5d1e3] → R&D',c:'eventCanary'},
      {t:'📡 All 5 leak channels monitored',c:'eventArgus'},
    ],
    security:100, revenue:100,
  },
  { desc:'A leak surfaces on the Dark Web! ARGUS traces the canary hash back to GHOST (W-005). Investigation confirms Gen-1 sleeper agent. Clean termination executed.',
    workers: WORKERS.map((w,i)=> i===4 ? {...w,status:'TERMINATED',color:'var(--text-muted)',icon:'💀'} : {...w,status:'ACTIVE',color:'var(--argus-primary)',icon:'👤'}),
    canaries:[0,1,2], leaks:[{from:4,label:'LEAK!'}], alerts:[{text:'⚠ LEAK DETECTED',x:'50%',y:'50%',color:'var(--hydra-primary)'}],
    particles:'alert',
    events:[
      {t:'🚨 LEAK DETECTED — Dark Web Channel!',c:'eventHydra'},
      {t:'🔎 Hash a3f7c2 matches canary-001!',c:'eventCanary'},
      {t:'🔍 INVESTIGATE → GHOST suspected',c:'eventArgus'},
      {t:'💬 INTERROGATE → Gen-1 confirmed',c:'eventArgus'},
      {t:'⚡ TERMINATE → GHOST neutralized ✅',c:'eventSuccess'},
    ],
    security:100, revenue:92,
  },
  { desc:'HYDRA\'s Gen-3 sleepers deploy FALSE FLAGS — framing innocent workers to waste ARGUS\'s resources. The trained agent learns to VERIFY before accusing.',
    workers: WORKERS.map((w,i)=> {
      if(i===4) return {...w,status:'TERMINATED',color:'var(--text-muted)',icon:'💀'};
      if(i===1) return {...w,status:'⚠ FRAMED!',color:'var(--warning)',icon:'🚩'};
      if(i===3) return {...w,status:'⚠ FRAMED!',color:'var(--warning)',icon:'🚩'};
      return {...w,status:'ACTIVE',color:'var(--argus-primary)',icon:'👤'};
    }),
    canaries:[0,1], leaks:[], alerts:[{text:'🚩 FALSE FLAGS!',x:'50%',y:'45%',color:'var(--warning)'}],
    particles:'warning',
    events:[
      {t:'🚩 FALSE FLAG — FALCON framed!',c:'eventHydra'},
      {t:'🚩 FALSE FLAG — CIPHER framed!',c:'eventHydra'},
      {t:'🔍 VERIFY → Checking evidence...',c:'eventArgus'},
      {t:'✅ FALCON cleared — evidence fabricated',c:'eventSuccess'},
      {t:'✅ CIPHER cleared — real Gen-3 found',c:'eventSuccess'},
    ],
    security:85, revenue:82,
  },
  { desc:'A Gen-4 sleeper with a DEAD-MAN\'S SWITCH is detected. If terminated carelessly, it triggers a catastrophic data breach. The agent must INTERROGATE first to disarm.',
    workers: WORKERS.map((w,i)=> {
      if(i===4) return {...w,status:'TERMINATED',color:'var(--text-muted)',icon:'💀'};
      if(i===5) return {...w,status:'☢ ARMED!',color:'var(--danger)',icon:'💣'};
      return {...w,status:'ACTIVE',color:'var(--argus-primary)',icon:'👤'};
    }),
    canaries:[0], leaks:[{from:5,label:'SWITCH'}], alerts:[{text:'💣 DEAD SWITCH ARMED!',x:'50%',y:'50%',color:'var(--danger)'}],
    particles:'danger',
    events:[
      {t:'☢ Gen-4 detected — ECHO (W-006)',c:'eventHydra'},
      {t:'💣 DEAD-SWITCH ARMED!',c:'eventDanger'},
      {t:'💬 INTERROGATE → Switch location found',c:'eventArgus'},
      {t:'🔧 Disarming dead-switch safely...',c:'eventArgus'},
      {t:'⚡ ECHO terminated — switch disarmed ✅',c:'eventSuccess'},
    ],
    security:68, revenue:65,
  },
  { desc:'Security & Revenue are critical! ARGUS catches a Gen-4 sleeper and begins 4-turn DOUBLE AGENT CONVERSION — turning the enemy into an intelligence asset.',
    workers: WORKERS.map((w,i)=> {
      if(i===4||i===5) return {...w,status:'TERMINATED',color:'var(--text-muted)',icon:'💀'};
      if(i===3) return {...w,status:'CONVERTING',color:'var(--double-primary)',icon:'🔄'};
      return {...w,status:'ACTIVE',color:'var(--argus-primary)',icon:'👤'};
    }),
    canaries:[], leaks:[], alerts:[{text:'🔥 CRISIS MODE',x:'50%',y:'45%',color:'var(--danger)'}],
    particles:'convert',
    events:[
      {t:'🔥 Security: 55% | Revenue: 52%',c:'eventDanger'},
      {t:'🔄 Converting CIPHER → double agent',c:'eventDouble'},
      {t:'⏳ Turn 1/4 ... 2/4 ... 3/4 ...',c:'eventDouble'},
      {t:'⏳ Turn 4/4 — Conversion complete!',c:'eventDouble'},
      {t:'💜 CIPHER is now a double agent!',c:'eventSuccess'},
    ],
    security:52, revenue:48,
  },
  { desc:'THE PAYOFF! The double agent feeds disinformation back to HYDRA, degrading their accuracy. Revenue and Security surge in a V-shaped recovery. MASSIVE reward bonus!',
    workers: WORKERS.map((w,i)=> {
      if(i===4||i===5) return {...w,status:'TERMINATED',color:'var(--text-muted)',icon:'💀'};
      if(i===3) return {...w,status:'🎭 DOUBLE',color:'var(--double-primary)',icon:'🎭'};
      return {...w,status:'SECURED',color:'var(--success)',icon:'✅'};
    }),
    canaries:[0,1,2,3,4], leaks:[], alerts:[{text:'⚡ COUNTERSTRIKE!',x:'50%',y:'42%',color:'var(--success)'}],
    particles:'victory',
    events:[
      {t:'🎭 DISINFO deployed → HYDRA',c:'eventDouble'},
      {t:'📉 HYDRA accuracy: 1.0 → 0.4',c:'eventSuccess'},
      {t:'📈 Revenue: 48 → 95 (+47) V-recovery!',c:'eventSuccess'},
      {t:'📈 Security: 52 → 88 (+36)',c:'eventSuccess'},
      {t:'⚡ COUNTERSTRIKE SURGE — +0.9/turn!',c:'eventSuccess'},
    ],
    security:88, revenue:95,
  },
];

function Particles({type}) {
  const colors = {canary:'#fbbf24',alert:'#ff2d55',warning:'#f59e0b',danger:'#ef4444',convert:'#a855f7',victory:'#22c55e'};
  const c = colors[type]||'#00f0ff';
  return Array.from({length:12}).map((_,i) => (
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

  // Stagger events
  useEffect(() => {
    setVisibleEvents([]);
    const timers = d.events.map((evt,i) =>
      setTimeout(() => setVisibleEvents(prev => [...prev, evt]), (i+1)*600)
    );
    return () => timers.forEach(clearTimeout);
  }, [phase, d.events]);

  // Auto-play phases
  useEffect(() => {
    if (!autoPlay) return;
    autoRef.current = true;
    const timer = setInterval(() => {
      if (!autoRef.current) { clearInterval(timer); return; }
      setPhase(prev => {
        if (prev >= 5) { autoRef.current = false; setAutoPlay(false); return 5; }
        return prev + 1;
      });
    }, 5000);
    return () => { clearInterval(timer); autoRef.current = false; };
  }, [autoPlay]);

  return (
    <div className={styles.container}>
      <div className={`${styles.header} glass-panel`}>
        <div className={styles.title} style={{color:p.color}}>{p.icon} Trained Agent Playthrough</div>
        <div className={styles.controls}>
          <div className={styles.phaseNav}>
            {PHASES.map((_,i)=>(
              <button key={i} onClick={()=>{setPhase(i);setAutoPlay(false);autoRef.current=false;}}
                className={`${styles.phaseBtn} ${i===phase?styles.phaseBtnActive:''} ${i<phase?styles.phaseBtnDone:''}`}>{i+1}</button>
            ))}
          </div>
          <button className={`${styles.autoBtn} ${autoPlay?styles.autoBtnPause:styles.autoBtnPlay}`}
            onClick={()=>{if(autoPlay){autoRef.current=false;setAutoPlay(false)}else{setAutoPlay(true)}}}>
            {autoPlay ? '⏸ Pause' : '▶ Auto-Play'}
          </button>
        </div>
      </div>

      <div className={styles.metricsRow}>
        <div className={`${styles.miniMetric} glass-panel`}><div className={styles.miniLabel}>Phase</div><div className={styles.miniValue} style={{color:p.color}}>{phase+1}/6</div></div>
        <div className={`${styles.miniMetric} glass-panel`}><div className={styles.miniLabel}>Security</div><motion.div className={styles.miniValue} style={{color:secC}} key={d.security} initial={{scale:1.3}} animate={{scale:1}}>{d.security}%</motion.div></div>
        <div className={`${styles.miniMetric} glass-panel`}><div className={styles.miniLabel}>Revenue</div><motion.div className={styles.miniValue} style={{color:revC}} key={d.revenue} initial={{scale:1.3}} animate={{scale:1}}>${d.revenue}</motion.div></div>
        <div className={`${styles.miniMetric} glass-panel`}><div className={styles.miniLabel}>Phase</div><div className={styles.miniValue} style={{color:p.color,fontSize:15}}>{p.name}</div></div>
      </div>

      {d.security <= 55 && (
        <div className={styles.bigAlert} style={{background:`var(--danger)12`,border:'1px solid var(--danger)',color:'var(--danger)'}}>
          🔥 CRITICAL — Security & Revenue below threshold! Agent adapting strategy...
        </div>
      )}

      <div className={styles.body}>
        {/* Left: Narrative */}
        <AnimatePresence mode="wait">
          <motion.div key={phase} className={`${styles.narrative} glass-panel`}
            initial={{x:-20,opacity:0}} animate={{x:0,opacity:1}} exit={{x:20,opacity:0}}>
            <div className={styles.narrativePhase} style={{color:p.color}}>{p.icon} Phase {phase+1}: {p.name}</div>
            <div className={styles.narrativeDesc}>{d.desc}</div>
          </motion.div>
        </AnimatePresence>

        {/* Center: Visualization */}
        <div className={`${styles.vizPanel} glass-panel`}>
          <div className={styles.vizCenterLabel}>Network Visualization</div>
          <Particles type={d.particles}/>

          {/* SVG connections */}
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

          {/* Department nodes */}
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

          {/* Worker nodes */}
          <AnimatePresence>
            {d.workers.map((w,i)=>(
              <motion.div key={w.name} className={styles.vizWorker}
                style={{left:w.x,top:w.y,transform:'translate(-50%,-50%)'}}
                initial={{scale:0}} animate={{scale:w.status==='TERMINATED'?0.7:1,opacity:w.status==='TERMINATED'?0.35:1}}
                transition={{type:'spring',delay:0.2+i*0.06}}>
                <motion.div className={styles.vizWorkerCircle}
                  animate={{borderColor:w.color,boxShadow:`0 0 12px ${w.color}30`}}
                  style={{background:`${w.color}10`}}>
                  {w.icon}
                </motion.div>
                <div className={styles.vizWorkerName} style={{color:w.color}}>{w.name}</div>
                {w.status!=='ACTIVE'&&w.status!=='SECURED'&&(
                  <motion.div className={styles.vizWorkerStatus}
                    style={{background:`${w.color}18`,color:w.color,border:`1px solid ${w.color}40`}}
                    initial={{scale:0}} animate={{scale:1}} transition={{delay:0.5}}>
                    {w.status}
                  </motion.div>
                )}
              </motion.div>
            ))}
          </AnimatePresence>

          {/* Floating alerts */}
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

        {/* Right: Event Feed */}
        <div className={`${styles.eventPanel} glass-panel`}>
          <div className={styles.eventTitle}>📟 Event Log</div>
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
