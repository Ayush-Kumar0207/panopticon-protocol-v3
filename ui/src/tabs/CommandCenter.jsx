import { useState, useRef, useEffect, useCallback } from 'react';
import { motion, AnimatePresence } from 'framer-motion';
import { getAgentStatus, resetEnvironment, stepWithAgent } from '../api/client';
import { DEPARTMENTS } from '../data/constants';
import styles from './CommandCenter.module.css';

const DP=[{x:'22%',y:'18%'},{x:'62%',y:'12%'},{x:'78%',y:'42%'},{x:'62%',y:'72%'},{x:'22%',y:'72%'},{x:'8%',y:'42%'}];
const DI=['⚙️','💰','🔬','🏭','👔','⚖️'];
const DV=['engineering','finance','rd','operations','executive','legal'];
const LV=['easy','medium','hard','level_4','level_5'];
const LL={easy:'Easy',medium:'Medium',hard:'Hard',level_4:'Lv.4',level_5:'Lv.5'};
const ROLES={engineering:'Engineer',finance:'Accountant',rd:'Researcher',operations:'Operator',executive:'Executive',legal:'Counsel'};
const SC={SPY_CATCH:{l:'🟢 Perfect Spy Catch',c:'var(--success)',d:'Canary→Verify→Interrogate→Terminate'},PARANOIA:{l:'🔴 False Accusation!',c:'var(--danger)',d:'Innocent terminated'},DEAD_SWITCH:{l:'💣 Dead Switch!',c:'var(--danger)',d:'Catastrophic damage'},COUNTERSTRIKE:{l:'⚡ Counterstrike!',c:'var(--success)',d:'Double agent deployed'},TURNING:{l:'🔄 Turning Agent',c:'var(--double-primary)',d:'Converting spy'}};

function getPhase(t,m){const p=t/Math.max(m,1);if(p<.15)return{n:'Orientation',i:'🛡️',c:'var(--argus-primary)'};if(p<.3)return{n:'First Contact',i:'📡',c:'var(--canary-primary)'};if(p<.5)return{n:'Escalation',i:'⚠️',c:'var(--warning)'};if(p<.7)return{n:'Deep Cover',i:'💣',c:'var(--danger)'};if(p<.85)return{n:'Crisis',i:'🔥',c:'var(--hydra-primary)'};return{n:'Counterstrike',i:'⚡',c:'var(--success)'};}

function roleName(w){
  if(w.state==='double_agent')return '🎭 Double Agent';
  if(w.state==='terminated')return '💀 Eliminated';
  if(w.state==='compromised')return '☠️ Burned';
  if(w.turning_in_progress)return '🔄 Converting...';
  if(w.state==='suspected'||w.suspicion_level>.5)return '⚠️ Suspect';
  return ROLES[w.department]||'Staff';
}
function wColor(w){const s=w?.state||'';if(s==='double_agent')return'var(--double-primary)';if(s==='terminated')return'var(--text-muted)';if(s==='suspected')return'var(--warning)';if(s==='compromised')return'var(--hydra-primary)';return'var(--argus-primary)';}
function wIcon(w){if(w.state==='terminated')return'💀';if(w.state==='double_agent')return'🎭';if(w.turning_in_progress)return'🔄';if(w.state==='suspected'||w.suspicion_level>.5)return'⚠️';if(w.state==='compromised')return'☠️';return'👤';}
function clsEvt(e){const l=e.toLowerCase();if(l.includes('canary'))return'eventCanary';if(l.includes('double')||l.includes('disinfo')||l.includes('turned'))return'eventDouble';if(l.includes('false')||l.includes('innocent')||l.includes('dead')||l.includes('game over'))return'eventHydra';if(l.includes('sleeper')||l.includes('hydra')||l.includes('leak'))return'eventHydra';if(l.includes('eliminated')||l.includes('caught')||l.includes('identified'))return'eventSuccess';return'eventArgus';}
function detectSc(evts){const j=evts.join(' ').toLowerCase();if(j.includes('dead')&&j.includes('switch')&&j.includes('triggered'))return'DEAD_SWITCH';if(j.includes('innocent')||j.includes('false accusation'))return'PARANOIA';if(j.includes('double agent acquired'))return'TURNING';if(j.includes('disinformation'))return'COUNTERSTRIKE';if(j.includes('eliminated'))return'SPY_CATCH';return null;}
function fmtAction(a){if(!a?.action_type)return'🧠 ARGUS thinking...';const t=a.action_type.toUpperCase();const target=a.target?` ${a.target}`:'';const sub=a.sub_action&&a.sub_action!=='none'?`/${a.sub_action}`:'';return`🧠 ${t}${sub}${target}`;}

function gradePerf(score,sec,rev,turns,maxTurns){
  const survived=turns>=maxTurns;
  if(sec>=90&&score>=3)return{grade:'🏆 EXCELLENT',color:'var(--success)',text:`Agent maintained ${sec.toFixed(0)}% security with +${score.toFixed(1)} reward. Flawless defense.`};
  if(sec>=70&&score>=1)return{grade:'✅ GOOD',color:'var(--argus-primary)',text:`Solid performance. Security held at ${sec.toFixed(0)}%, revenue $${rev.toFixed(0)}.`};
  if(sec>=40&&score>=0)return{grade:'⚠️ ADEQUATE',color:'var(--warning)',text:`Agent struggled. Security dropped to ${sec.toFixed(0)}%. Room for improvement.`};
  if(!survived)return{grade:'💀 FAILED',color:'var(--danger)',text:`Mission ended early! ${sec<=0?'Total security breach.':'Enterprise went bankrupt.'}`};
  return{grade:'❌ POOR',color:'var(--danger)',text:`Negative reward (${score.toFixed(1)}). Agent failed to counter HYDRA effectively.`};
}

export default function CommandCenter({serverOnline}){
  const[level,setLevel]=useState('easy');
  const[obs,setObs]=useState(null);
  const[turn,setTurn]=useState(0);
  const[maxTurns,setMaxTurns]=useState(60);
  const[reward,setReward]=useState(0);
  const[totalReward,setTotalReward]=useState(0);
  const[done,setDone]=useState(false);
  const[loading,setLoading]=useState(false);
  const[events,setEvents]=useState([]);
  const[autoPlaying,setAutoPlaying]=useState(false);
  const[currentAction,setCurrentAction]=useState(null);
  const[activeScenario,setActiveScenario]=useState(null);
  const[speed,setSpeed]=useState(1);
  const[stepCount,setStepCount]=useState(0);
  const[popups,setPopups]=useState([]);
  const[agentStatus,setAgentStatus]=useState({status:'unknown',policy:'trained',model_ref:'trained_model',loaded:false,local_model_present:false,error:null});
  const autoRef=useRef(false);
  const stepRef=useRef(0);
  const obsRef=useRef(null);
  const histRef=useRef([]);
  const totalRef=useRef(0);
  const eventEndRef=useRef(null);

  const addEvt=useCallback((t,c='eventSystem')=>setEvents(p=>[...p.slice(-80),{text:t,type:c,ts:Date.now()}]),[]);
  const addPopup=useCallback((text,color,x,y)=>{
    const id=Date.now()+Math.random();
    setPopups(p=>[...p,{id,text,color,x,y}]);
    setTimeout(()=>setPopups(p=>p.filter(pp=>pp.id!==id)),3000);
  },[]);

  useEffect(()=>{if(eventEndRef.current)eventEndRef.current.scrollIntoView({behavior:'smooth'});},[events]);

  useEffect(()=>{
    if(!serverOnline){
      setAgentStatus(p=>({...p,status:'offline'}));
      return;
    }
    let mounted=true;
    const loadStatus=async()=>{
      try{
        const data=await getAgentStatus();
        if(mounted)setAgentStatus(data);
      }catch(e){
        if(mounted)setAgentStatus({status:'error',policy:'trained',model_ref:'trained_model',loaded:false,local_model_present:false,error:e.message});
      }
    };
    loadStatus();
    return()=>{mounted=false;};
  },[serverOnline]);

  const handleReset=async()=>{
    setLoading(true);setEvents([]);setTotalReward(0);totalRef.current=0;setDone(false);setStepCount(0);stepRef.current=0;setCurrentAction(null);setActiveScenario(null);histRef.current=[];setPopups([]);
    try{
      const d=await resetEnvironment(level);
      const o=d.observation;setObs(o);obsRef.current=o;setTurn(o.turn||0);setMaxTurns(o.max_turns||160);setReward(0);
      addEvt(`━━━ MISSION INITIATED ━━━`,'eventArgus');
      addEvt(`🎯 ${level.toUpperCase()} | Workers: ${(o.workers||[]).length} | Turns: ${o.max_turns||160}`,'eventSystem');
      addEvt(`🧠 Control policy: trained model (${agentStatus.loaded?'loaded':'lazy-load'})`,'eventArgus');
    }catch(e){
      addEvt(`❌ ${e.message}`,'eventHydra');
    }
    setLoading(false);return true;
  };

  const executeStep=useCallback(async()=>{
    if(!autoRef.current||!obsRef.current)return false;
    try{
      setCurrentAction('🧠 ARGUS model inference...');
      const data=await stepWithAgent();
      const o=data.observation;setObs(o);obsRef.current=o;setTurn(o.turn||0);setReward(data.reward);totalRef.current+=data.reward;setTotalReward(totalRef.current);stepRef.current+=1;setStepCount(stepRef.current);
      const action=data.agent_action||data.info?.agent_action||null;
      const actionLabel=fmtAction(action);
      setCurrentAction(actionLabel);
      if(action?.action_type)histRef.current.push(action.action_type);
      const actionReason=action?.reason?` — ${action.reason}`:'';
      addEvt(`${actionLabel}${actionReason}`,'eventArgus');
      if(data.model_info){
        setAgentStatus(p=>({...p,status:'ready',loaded:true,model_ref:data.model_info.model_ref||p.model_ref,error:null}));
      }
      const be=data.info?.events||[];
      be.forEach(e=>{
        if(e.startsWith('[HIDDEN]'))return;
        addEvt(e,clsEvt(e));
        const el=e.toLowerCase();
        if(el.includes('canary')&&el.includes('planted'))addPopup('🪤 Canary Planted!','var(--canary-primary)',`${30+Math.random()*40}%`,`${20+Math.random()*40}%`);
        if(el.includes('leak')&&el.includes('detected'))addPopup('🚨 Leak Detected!','var(--hydra-primary)',`${40+Math.random()*30}%`,`${30+Math.random()*30}%`);
        if(el.includes('eliminated')||el.includes('terminated'))addPopup('⚡ Threat Eliminated!','var(--success)',`${35+Math.random()*30}%`,`${25+Math.random()*30}%`);
        if(el.includes('double agent acquired'))addPopup('🎭 Agent Turned!','var(--double-primary)',`${40+Math.random()*20}%`,`${40+Math.random()*20}%`);
        if(el.includes('dead')&&el.includes('switch'))addPopup('💣 DEAD SWITCH!','var(--danger)',`${35+Math.random()*30}%`,`${30+Math.random()*20}%`);
        if(el.includes('false flag'))addPopup('🚩 False Flag!','var(--warning)',`${30+Math.random()*40}%`,`${30+Math.random()*30}%`);
        if(el.includes('innocent'))addPopup('❌ Innocent Fired!','var(--danger)',`${35+Math.random()*30}%`,`${35+Math.random()*20}%`);
      });
      const sc=detectSc(be);if(sc)setActiveScenario(SC[sc]);
      const rStr=data.reward>=0?`+${data.reward.toFixed(3)}`:data.reward.toFixed(3);
      addEvt(`📊 ${rStr} | Sec: ${o.security_score?.toFixed(0)}% | Rev: $${o.enterprise_revenue?.toFixed(0)}`,data.reward>=.05?'eventSuccess':data.reward>=-.05?'eventSystem':'eventHydra');
      if(data.done||data.truncated){
        setDone(true);autoRef.current=false;setAutoPlaying(false);setCurrentAction(null);
        return false;
      }
      return true;
    }catch(e){
      autoRef.current=false;setAutoPlaying(false);setCurrentAction(null);
      setAgentStatus(p=>({...p,status:'error',error:e.message}));
      addEvt(`❌ Trained model step failed: ${e.message}`,'eventHydra');
      return false;
    }
  },[addEvt,addPopup]);

  useEffect(()=>{if(!autoPlaying||done)return;autoRef.current=true;const dl=2000/speed;let t;const lp=async()=>{if(!autoRef.current)return;const c=await executeStep();if(c&&autoRef.current)t=setTimeout(lp,dl);};t=setTimeout(lp,400);return()=>clearTimeout(t);},[autoPlaying,done,speed,executeStep]);

  const handleStart=async()=>{if(!obs)await handleReset();setTimeout(()=>{autoRef.current=true;setAutoPlaying(true);},600);};
  const handlePause=()=>{autoRef.current=false;setAutoPlaying(false);setCurrentAction(null);};
  const resetAll=()=>{setObs(null);setDone(false);setEvents([]);setTotalReward(0);totalRef.current=0;setStepCount(0);stepRef.current=0;obsRef.current=null;histRef.current=[];setActiveScenario(null);setPopups([]);};

  const workers=obs?.workers||[];
  const security=obs?.security_score??100;
  const revenue=obs?.enterprise_revenue??100;
  const phase=getPhase(turn,maxTurns);
  const canaries=obs?.canary_traps?.filter(c=>c.active)?.length??0;
  const leaks=obs?.active_leaks?.length??0;
  const doubles=obs?.double_agents?.filter(d=>d.active)?.length??0;
  const secC=security>70?'var(--success)':security>40?'var(--warning)':'var(--danger)';
  const revC=revenue>80?'var(--success)':revenue>50?'var(--warning)':'var(--danger)';
  const perf=done?gradePerf(totalReward,security,revenue,turn,maxTurns):null;
  const modelStatusColor=agentStatus.status==='ready'?'var(--success)':agentStatus.status==='configured'?'var(--warning)':agentStatus.status==='offline'?'var(--danger)':'var(--danger)';
  const modelStatusLabel=agentStatus.status==='ready'?'TRAINED READY':agentStatus.status==='configured'?'MODEL STANDBY':agentStatus.status==='offline'?'BACKEND OFFLINE':'MODEL ERROR';
  const modelRefLabel=agentStatus.model_ref?(agentStatus.model_ref.includes('\\')||agentStatus.model_ref.includes('/')?agentStatus.model_ref.split(/[/\\]/).slice(-2).join('/') : agentStatus.model_ref):'trained_model';

  return(
    <div className={styles.container}>
      {!serverOnline&&<div className={styles.offlineOverlay}><div className={styles.offlineTitle}>⚡ Backend Offline</div><div className={styles.offlineDesc}>Run: <code>.venv-infer\Scripts\python.exe _server.py</code></div></div>}

      <div className={styles.gameHeader}>
        <div className={styles.levelSelect}>{LV.map(l=>(<button key={l} className={`${styles.levelBtn} ${level===l?styles.levelBtnActive:''}`} onClick={()=>{if(!autoPlaying){setLevel(l);resetAll();}}} disabled={autoPlaying}>{LL[l]}</button>))}</div>
        <div className={styles.turnInfo}>TURN <span>{turn}</span>/{maxTurns} {autoPlaying&&<span className={styles.liveTag}>● LIVE</span>}</div>
        <div className={styles.controlGroup}>
          {autoPlaying?<button className={styles.pauseBtn} onClick={handlePause}>⏸ PAUSE</button>:done?<button className={styles.resetBtn} onClick={resetAll}>⟳ NEW MISSION</button>:<button className={styles.startBtn} onClick={handleStart} disabled={loading||!serverOnline}>▶ {obs?'RESUME MODEL':'START MODEL'}</button>}
          <div className={styles.speedControl}>{[1,2,3].map(s=>(<button key={s} className={`${styles.speedBtn} ${speed===s?styles.speedBtnActive:''}`} onClick={()=>setSpeed(s)}>{s}x</button>))}</div>
        </div>
      </div>

      <div className={`${styles.eventLog} glass-panel`}>
        <div className={styles.eventLogTitle}>📟 Trained-Agent Feed {autoPlaying&&<span className={styles.liveTag}>● LIVE</span>}</div>
        <div className={styles.events}>
          {events.length===0&&<div className={styles.event} style={{borderColor:'var(--text-muted)',color:'var(--text-muted)'}}>Press ▶ START MODEL</div>}
          <AnimatePresence>{events.map((e,i)=>(<motion.div key={e.ts+'-'+i} className={`${styles.event} ${styles[e.type]}`} initial={{x:-20,opacity:0}} animate={{x:0,opacity:1}} transition={{duration:.12}}>{e.text}</motion.div>))}</AnimatePresence>
          <div ref={eventEndRef}/>
        </div>
      </div>

      <div className={`${styles.gameArea} glass-panel`}>
        <div className={styles.networkArea}>
          {!obs?(
            <motion.div className={styles.placeholder} initial={{opacity:0}} animate={{opacity:1}}>
              <div className={styles.placeholderIcon}>👁️</div>
              <div className={styles.placeholderTitle}>ARGUS DEFENSE MATRIX</div>
              <div className={styles.placeholderSub}>Select difficulty & press START MODEL to let the fine-tuned agent operate live</div>
            </motion.div>
          ):(
            <>
              <svg style={{position:'absolute',inset:0,width:'100%',height:'100%',pointerEvents:'none'}}>
                {[[0,1],[1,2],[2,3],[3,4],[4,5],[5,0],[0,3],[1,4],[2,5]].map(([a,b],i)=>(
                  <line key={i} x1={DP[a].x} y1={DP[a].y} x2={DP[b].x} y2={DP[b].y} stroke={leaks>0?'rgba(255,45,85,0.12)':'rgba(0,240,255,0.08)'} strokeWidth="1"/>
                ))}
              </svg>
              {DEPARTMENTS.map((dept,i)=>{
                const hc=(obs.canary_traps||[]).some(c=>c.department===DV[i]&&c.active);
                const hl=(obs.active_leaks||[]).some(l=>l.department===DV[i]);
                return(
                  <motion.div key={dept} className={styles.graphNode} style={{left:DP[i].x,top:DP[i].y,transform:'translate(-50%,-50%)'}} initial={{scale:0}} animate={{scale:1}} transition={{delay:i*.05,type:'spring'}}>
                    <motion.div className={styles.nodeCircle} animate={{borderColor:hl?'var(--hydra-primary)':hc?'var(--canary-primary)':'var(--argus-dim)',boxShadow:hl?'0 0 20px var(--hydra-glow)':hc?'0 0 15px rgba(251,191,36,0.3)':'0 0 8px var(--argus-glow)'}}>{DI[i]}</motion.div>
                    <span className={styles.nodeLabel}>{dept}</span>
                    {hc&&<motion.div className={styles.canaryBadge} initial={{scale:0}} animate={{scale:1}}>🪤</motion.div>}
                    {hl&&<motion.div className={styles.canaryBadge} style={{left:-8}} animate={{scale:[1,1.3,1]}} transition={{repeat:Infinity,duration:.8}}>🚨</motion.div>}
                  </motion.div>
                );
              })}
              {workers.slice(0,10).map((w,i)=>{
                const a=(i/Math.max(workers.length,8))*Math.PI*2-Math.PI/2;
                const cx=50+32*Math.cos(a),cy=50+32*Math.sin(a);
                const c=wColor(w),dead=w.state==='terminated';
                return(
                  <motion.div key={w.id||i} style={{position:'absolute',left:`${cx}%`,top:`${cy}%`,transform:'translate(-50%,-50%)',display:'flex',flexDirection:'column',alignItems:'center',gap:2,opacity:dead?.3:1}} initial={{scale:0}} animate={{scale:dead?.7:1}} transition={{type:'spring',delay:.3+i*.04}}>
                    <motion.div animate={{borderColor:c,boxShadow:w.turning_in_progress?'0 0 20px var(--double-primary)':`0 0 10px ${c}30`}} style={{width:30,height:30,borderRadius:'50%',border:'2px solid',background:`${c}12`,display:'flex',alignItems:'center',justifyContent:'center',fontSize:13}}>{wIcon(w)}</motion.div>
                    <span style={{fontFamily:'var(--font-mono)',fontSize:7,color:c}}>{(w.name||w.id).slice(0,7)}</span>
                    <span style={{fontFamily:'var(--font-mono)',fontSize:6,color:c,opacity:.7}}>{roleName(w)}</span>
                  </motion.div>
                );
              })}
              <AnimatePresence>{popups.map(p=>(<motion.div key={p.id} className={styles.floatingPopup} style={{left:p.x,top:p.y,background:`${p.color}20`,border:`1px solid ${p.color}`,color:p.color}} initial={{scale:0,opacity:0,y:10}} animate={{scale:1,opacity:1,y:0}} exit={{opacity:0,y:-20}} transition={{duration:.3}}>{p.text}</motion.div>))}</AnimatePresence>
            </>
          )}
        </div>
        <AnimatePresence>{currentAction&&(<motion.div className={styles.actionBanner} initial={{y:10,opacity:0}} animate={{y:0,opacity:1}} exit={{y:-10,opacity:0}}><div className={styles.actionPulse}/>{currentAction}</motion.div>)}</AnimatePresence>
        <AnimatePresence>{activeScenario&&(<motion.div className={styles.scenarioBanner} style={{borderColor:activeScenario.c,color:activeScenario.c}} initial={{x:-20,opacity:0}} animate={{x:0,opacity:1}} exit={{opacity:0}}><div style={{fontWeight:700}}>{activeScenario.l}</div><div style={{fontSize:10,opacity:.8}}>{activeScenario.d}</div></motion.div>)}</AnimatePresence>
        {done&&perf&&(<motion.div className={styles.gameOverBanner} style={{borderColor:perf.color,color:perf.color,background:`${perf.color}12`}} initial={{scale:.8,opacity:0}} animate={{scale:1,opacity:1}}>{perf.grade} — Score: {totalReward.toFixed(2)} | Security: {security.toFixed(0)}%<div className={styles.gradeText}>{perf.text}</div></motion.div>)}
      </div>

      <div className={styles.metricsPanel}>
        <div className={styles.phaseIndicator} style={{borderColor:phase.c,background:`${phase.c}10`}}><div style={{fontSize:20}}>{phase.i}</div><div className={styles.phaseName} style={{color:phase.c}}>{phase.n}</div></div>
        <div className={styles.metricCard}><div className={styles.metricLabel}>Model Status</div><div className={styles.metricValue} style={{color:modelStatusColor,fontSize:18}}>{modelStatusLabel}</div><div className={styles.metricSub} style={{wordBreak:'break-all'}}>{modelRefLabel}</div>{agentStatus.error&&<div className={styles.metricSub} style={{color:'var(--danger)'}}>{agentStatus.error}</div>}</div>
        <div className={styles.metricCard}><div className={styles.metricLabel}>🛡️ Security (55%)</div><motion.div className={styles.metricValue} style={{color:secC}} key={Math.round(security)} initial={{scale:1.1}} animate={{scale:1}}>{security.toFixed(0)}%</motion.div><div className={styles.metricBar}><motion.div className={styles.metricBarFill} animate={{width:`${Math.max(0,security)}%`,background:secC}}/></div></div>
        <div className={styles.metricCard}><div className={styles.metricLabel}>💰 Revenue (45%)</div><motion.div className={styles.metricValue} style={{color:revC}} key={Math.round(revenue)} initial={{scale:1.1}} animate={{scale:1}}>${revenue.toFixed(0)}</motion.div><div className={styles.metricBar}><motion.div className={styles.metricBarFill} animate={{width:`${Math.min(100,(revenue/150)*100)}%`,background:revC}}/></div></div>
        <div className={styles.metricCard}><div className={styles.metricLabel}>Cumulative Reward</div><div className={styles.rewardAccum} style={{color:totalReward>=0?'var(--argus-primary)':'var(--hydra-primary)'}}>{totalReward>=0?'+':''}{totalReward.toFixed(2)}</div></div>
        <div className={styles.metricCard}><div className={styles.metricLabel}>Last Step</div><motion.div className={styles.metricValue} style={{color:reward>=0?'var(--success)':'var(--danger)',fontSize:18}} key={reward} initial={{scale:1.2}} animate={{scale:1}}>{reward>=0?'+':''}{reward.toFixed(3)}</motion.div></div>
        <div className={styles.metricCard}><div className={styles.metricLabel}>Intel Status</div><div className={styles.metricSub}>🪤 Canaries: {canaries}</div><div className={styles.metricSub}>⚠ Leaks: {leaks}</div><div className={styles.metricSub}>🎭 Double Agents: {doubles}</div><div className={styles.metricSub}>📊 Phase: {obs?.phase_number||1}/6</div><div className={styles.metricSub}>📋 Steps: {stepCount}</div></div>
        {workers.length>0&&(<div className={styles.metricCard}><div className={styles.metricLabel}>Workforce ({workers.filter(w=>w.state!=='terminated').length}/{workers.length})</div><div className={styles.workerList}>{workers.slice(0,12).map((w,i)=>{const c=wColor(w);return(<div key={i} className={styles.workerRow}><div className={styles.workerDot} style={{background:c}}/><span>{(w.name||w.id).slice(0,7)}</span><span className={styles.workerRole} style={{color:c}}>{roleName(w)}</span></div>);})}</div></div>)}
      </div>
    </div>
  );
}
