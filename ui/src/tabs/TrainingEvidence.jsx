import { useState } from 'react';
import { motion } from 'framer-motion';
import {
  AreaChart, Area, LineChart, Line, BarChart, Bar, RadarChart, Radar,
  PolarGrid, PolarAngleAxis, PolarRadiusAxis,
  XAxis, YAxis, CartesianGrid, Tooltip, Legend, ResponsiveContainer,
} from 'recharts';
import trainingData from '../data/trainingData.json';
import styles from './TrainingEvidence.module.css';

const COLORS = { easy:'#22c55e', medium:'#eab308', hard:'#f97316', level_4:'#ef4444', level_5:'#a855f7' };
const LEVEL_KEYS = Object.keys(COLORS);

function buildCurve(metricKey) {
  const steps = trainingData.easy?.skill ?? [];
  return steps.map((skill, index) => (
    LEVEL_KEYS.reduce((row, levelKey) => {
      const levelData = trainingData[levelKey] ?? {};
      row[levelKey] = levelData[metricKey]?.[index] ?? 0;
      return row;
    }, { step: Math.round(skill * 100) })
  ));
}

const REWARD_DATA = buildCurve('reward_mean');
const SECURITY_DATA = buildCurve('security_mean');
const BAR_DATA = buildCurve('caught_mean');

const RADAR_DATA = [
  { dim:'Grade', rawV5:70, supervisor:79, random:65 },
  { dim:'Reward', rawV5:34, supervisor:100, random:0 },
  { dim:'Revenue', rawV5:68, supervisor:80, random:30 },
  { dim:'Security', rawV5:89, supervisor:100, random:69 },
  { dim:'Caught', rawV5:89, supervisor:100, random:92 },
];

const COMPARISON = [
  { metric:'Macro Grade', base:'0.641', raw:'0.702', supervisor:'0.790', icon:'Score' },
  { metric:'Acceptance Gate', base:'Reference', raw:'Failed', supervisor:'Passed', icon:'Gate' },
  { metric:'Level 4 Pass Rate', base:'-', raw:'50%', supervisor:'100%', icon:'L4' },
  { metric:'Level 5 Pass Rate', base:'-', raw:'5%', supervisor:'100%', icon:'L5' },
  { metric:'Level 5 Security', base:'84.8', raw:'60.47', supervisor:'100.0', icon:'Sec' },
  { metric:'Level 5 Sleepers Caught', base:'4.50', raw:'3.90', supervisor:'5.00', icon:'Catch' },
];

const SCENARIO_STATS = [
  { name:'Accepted Gate', supervisor:'100%', raw:'0%', desc:'Raw V5 failed; security-first supervisor passed', color:'var(--success)' },
  { name:'Level 4 Pass Rate', supervisor:'100%', raw:'50%', desc:'Supervisor removes the remaining missed-sleeper failures', color:'var(--argus-primary)' },
  { name:'Level 5 Pass Rate', supervisor:'100%', raw:'5%', desc:'Raw V5 still breaks on Manchurian difficulty', color:'var(--canary-primary)' },
  { name:'Level 4 Security', supervisor:'100%', raw:'86%', desc:'Supervisor keeps the security floor intact', color:'var(--double-primary)' },
  { name:'Level 5 Security', supervisor:'100%', raw:'60%', desc:'Raw V5 loses too much security without control logic', color:'var(--success)' },
];

export default function TrainingEvidence() {
  const [tab, setTab] = useState('curves');

  return (
    <div className={styles.container}>
      <div className={`${styles.header} glass-panel`}>
        <div className={styles.title}>Training Evidence - Security-First V5</div>
        <div className={styles.subtitle}>
          Compact diagnostics from Drive metadata and completed benchmark reports.
          <span className={styles.dataNote}> Raw V5 improved, but only the supervisor diagnostic passed acceptance.</span>
        </div>
      </div>

      <div className={styles.tabRow}>
        {['curves','comparison','scenarios'].map(t=>(
          <button key={t} className={`${styles.tabBtn} ${tab===t?styles.tabBtnActive:''}`} onClick={()=>setTab(t)}>
            {t==='curves'?'V5 Curves':t==='comparison'?'Gate Comparison':'Acceptance'}
          </button>
        ))}
      </div>

      {tab==='curves' && (
        <>
          <div className={styles.legend}>
            {Object.entries(COLORS).map(([k,c])=>(
              <span key={k} className={styles.legendItem}><span className={styles.legendDot} style={{background:c}}/>{k.replace('_',' ')}</span>
            ))}
          </div>
          <div className={styles.chartGrid}>
            <div className={`${styles.chartPanel} glass-panel`}>
              <div className={styles.chartTitle}>Episode Reward Curves</div>
              <ResponsiveContainer width="100%" height={220}>
                <AreaChart data={REWARD_DATA}>
                  <CartesianGrid strokeDasharray="3 3" stroke="rgba(255,255,255,0.05)"/>
                  <XAxis dataKey="step" stroke="#666" fontSize={10} label={{value:'Training %',position:'insideBottom',offset:-5,fontSize:10,fill:'#888'}}/>
                  <YAxis stroke="#666" fontSize={10} label={{value:'Reward',angle:-90,position:'insideLeft',fontSize:10,fill:'#888'}}/>
                  <Tooltip contentStyle={{background:'rgba(10,12,18,0.95)',border:'1px solid rgba(255,255,255,0.1)',fontSize:11}}/>
                  {Object.entries(COLORS).map(([k,c])=>(
                    <Area key={k} type="monotone" dataKey={k} stroke={c} fill={c} fillOpacity={0.1} strokeWidth={2}/>
                  ))}
                </AreaChart>
              </ResponsiveContainer>
            </div>

            <div className={`${styles.chartPanel} glass-panel`}>
              <div className={styles.chartTitle}>Security Score</div>
              <ResponsiveContainer width="100%" height={220}>
                <LineChart data={SECURITY_DATA}>
                  <CartesianGrid strokeDasharray="3 3" stroke="rgba(255,255,255,0.05)"/>
                  <XAxis dataKey="step" stroke="#666" fontSize={10}/>
                  <YAxis stroke="#666" fontSize={10} domain={[0,100]}/>
                  <Tooltip contentStyle={{background:'rgba(10,12,18,0.95)',border:'1px solid rgba(255,255,255,0.1)',fontSize:11}}/>
                  {Object.entries(COLORS).map(([k,c])=>(
                    <Line key={k} type="monotone" dataKey={k} stroke={c} strokeWidth={2} dot={false}/>
                  ))}
                </LineChart>
              </ResponsiveContainer>
            </div>

            <div className={`${styles.chartPanel} glass-panel`}>
              <div className={styles.chartTitle}>Sleepers Caught per Episode</div>
              <ResponsiveContainer width="100%" height={220}>
                <BarChart data={BAR_DATA}>
                  <CartesianGrid strokeDasharray="3 3" stroke="rgba(255,255,255,0.05)"/>
                  <XAxis dataKey="step" stroke="#666" fontSize={10}/>
                  <YAxis stroke="#666" fontSize={10}/>
                  <Tooltip contentStyle={{background:'rgba(10,12,18,0.95)',border:'1px solid rgba(255,255,255,0.1)',fontSize:11}}/>
                  {Object.entries(COLORS).map(([k,c])=>(
                    <Bar key={k} dataKey={k} fill={c} fillOpacity={0.7} radius={[2,2,0,0]}/>
                  ))}
                </BarChart>
              </ResponsiveContainer>
            </div>

            <div className={`${styles.chartPanel} glass-panel`}>
              <div className={styles.chartTitle}>V5 Benchmark Radar - Raw vs Supervisor</div>
              <ResponsiveContainer width="100%" height={220}>
                <RadarChart data={RADAR_DATA}>
                  <PolarGrid stroke="rgba(255,255,255,0.1)"/>
                  <PolarAngleAxis dataKey="dim" stroke="#888" fontSize={10}/>
                  <PolarRadiusAxis angle={90} domain={[0,100]} stroke="#555" fontSize={8}/>
                  <Radar name="Raw V5" dataKey="rawV5" stroke="#00f0ff" fill="#00f0ff" fillOpacity={0.18} strokeWidth={2}/>
                  <Radar name="Supervisor" dataKey="supervisor" stroke="#22c55e" fill="#22c55e" fillOpacity={0.18} strokeWidth={2}/>
                  <Radar name="Random Agent" dataKey="random" stroke="#ff2d55" fill="#ff2d55" fillOpacity={0.08} strokeWidth={1} strokeDasharray="4 4"/>
                  <Legend wrapperStyle={{fontSize:10}}/>
                </RadarChart>
              </ResponsiveContainer>
            </div>
          </div>
        </>
      )}

      {tab==='comparison' && (
        <div className={styles.comparisonSection}>
          <div className={styles.comparisonTitle}>Base vs Raw V5 vs Supervisor - Acceptance Side by Side</div>
          <div className={styles.comparisonGrid}>
            <div className={styles.comparisonHeader}>
              <span>Metric</span><span style={{color:'var(--hydra-primary)'}}>Base/Ref</span><span style={{color:'var(--argus-primary)'}}>Raw V5</span><span>Supervisor</span>
            </div>
            {COMPARISON.map((c,i)=>(
              <motion.div key={i} className={`${styles.comparisonRow} glass-panel`}
                initial={{x:-20,opacity:0}} animate={{x:0,opacity:1}} transition={{delay:i*0.1}}>
                <span className={styles.compMetric}>{c.icon} {c.metric}</span>
                <span className={styles.compRandom}>{c.base}</span>
                <span className={styles.compTrained}>{c.raw}</span>
                <span className={styles.compImprove}>{c.supervisor}</span>
              </motion.div>
            ))}
          </div>
          <div className={styles.radarCompare}>
            <ResponsiveContainer width="100%" height={280}>
              <RadarChart data={RADAR_DATA}>
                <PolarGrid stroke="rgba(255,255,255,0.1)"/>
                <PolarAngleAxis dataKey="dim" stroke="#888" fontSize={11}/>
                <PolarRadiusAxis angle={90} domain={[0,100]} stroke="#555" fontSize={9}/>
                <Radar name="Raw V5" dataKey="rawV5" stroke="#00f0ff" fill="#00f0ff" fillOpacity={0.18} strokeWidth={2}/>
                <Radar name="Supervisor" dataKey="supervisor" stroke="#22c55e" fill="#22c55e" fillOpacity={0.18} strokeWidth={2}/>
                <Radar name="Random Agent" dataKey="random" stroke="#ff2d55" fill="#ff2d55" fillOpacity={0.08} strokeWidth={1} strokeDasharray="4 4"/>
                <Legend wrapperStyle={{fontSize:11}}/>
              </RadarChart>
            </ResponsiveContainer>
          </div>
        </div>
      )}

      {tab==='scenarios' && (
        <div className={styles.scenarioSection}>
          <div className={styles.comparisonTitle}>Acceptance Recovery - Raw V5 vs Security-First Supervisor</div>
          <div className={styles.scenarioList}>
            {SCENARIO_STATS.map((s,i)=>(
              <motion.div key={i} className={`${styles.scenarioCard} glass-panel`}
                initial={{y:20,opacity:0}} animate={{y:0,opacity:1}} transition={{delay:i*0.12}}>
                <div className={styles.scenarioName} style={{color:s.color}}>{s.name}</div>
                <div className={styles.scenarioDesc}>{s.desc}</div>
                <div className={styles.scenarioBars}>
                  <div className={styles.scenarioBarRow}>
                    <span className={styles.scenarioBarLabel} style={{color:'var(--hydra-primary)'}}>Raw V5</span>
                    <div className={styles.scenarioBar}>
                      <motion.div className={styles.scenarioBarFill} style={{background:'var(--hydra-primary)'}} initial={{width:0}} animate={{width:s.raw}} transition={{delay:0.3+i*0.1,duration:0.8}}/>
                    </div>
                    <span className={styles.scenarioBarVal} style={{color:'var(--hydra-primary)'}}>{s.raw}</span>
                  </div>
                  <div className={styles.scenarioBarRow}>
                    <span className={styles.scenarioBarLabel} style={{color:s.color}}>Supervisor</span>
                    <div className={styles.scenarioBar}>
                      <motion.div className={styles.scenarioBarFill} style={{background:s.color}} initial={{width:0}} animate={{width:s.supervisor}} transition={{delay:0.5+i*0.1,duration:0.8}}/>
                    </div>
                    <span className={styles.scenarioBarVal} style={{color:s.color}}>{s.supervisor}</span>
                  </div>
                </div>
              </motion.div>
            ))}
          </div>
        </div>
      )}

      <div className={styles.statsRow}>
        <div className={`${styles.statCard} glass-panel`}><div className={styles.statLabel}>Raw V5 Grade</div><div className={styles.statValue} style={{color:'var(--argus-primary)'}}>0.7016</div><div className={styles.statSub}>Better than base, gate failed</div></div>
        <div className={`${styles.statCard} glass-panel`}><div className={styles.statLabel}>Supervisor Grade</div><div className={styles.statValue} style={{color:'var(--success)'}}>0.7905</div><div className={styles.statSub}>Accepted diagnostic</div></div>
        <div className={`${styles.statCard} glass-panel`}><div className={styles.statLabel}>Raw L5 Security</div><div className={styles.statValue} style={{color:'var(--warning)'}}>60.47</div><div className={styles.statSub}>Main failure mode</div></div>
        <div className={`${styles.statCard} glass-panel`}><div className={styles.statLabel}>V5 Examples</div><div className={styles.statValue} style={{color:'var(--double-primary)'}}>88,896</div><div className={styles.statSub}>Across 250 episodes</div></div>
      </div>
    </div>
  );
}
