import { useState, useEffect, useCallback } from 'react';

const TOTAL_SCENARIOS = 8;
const SIM_SUB_PHASES = 6;

export function useScenario() {
  const [scenarioIndex, setScenarioIndex] = useState(0);
  const [simPhase, setSimPhase] = useState(0);

  const isSimulation = scenarioIndex === 4;

  const next = useCallback(() => {
    if (isSimulation && simPhase < SIM_SUB_PHASES - 1) {
      setSimPhase(p => p + 1);
    } else {
      if (scenarioIndex < TOTAL_SCENARIOS - 1) {
        setScenarioIndex(i => i + 1);
        setSimPhase(0);
      }
    }
  }, [scenarioIndex, simPhase, isSimulation]);

  const prev = useCallback(() => {
    if (isSimulation && simPhase > 0) {
      setSimPhase(p => p - 1);
    } else {
      if (scenarioIndex > 0) {
        setScenarioIndex(i => i - 1);
        setSimPhase(0);
      }
    }
  }, [scenarioIndex, simPhase, isSimulation]);

  const goTo = useCallback((index) => {
    setScenarioIndex(Math.max(0, Math.min(index, TOTAL_SCENARIOS - 1)));
    setSimPhase(0);
  }, []);

  useEffect(() => {
    const handler = (e) => {
      if (e.key === 'ArrowRight' || e.key === ' ') {
        e.preventDefault();
        next();
      } else if (e.key === 'ArrowLeft') {
        e.preventDefault();
        prev();
      } else if (e.key === 'Home') {
        e.preventDefault();
        goTo(0);
      } else if (e.key === 'End') {
        e.preventDefault();
        goTo(TOTAL_SCENARIOS - 1);
      } else if (e.key === 'f' || e.key === 'F') {
        if (!document.fullscreenElement) {
          document.documentElement.requestFullscreen();
        } else {
          document.exitFullscreen();
        }
      }
    };
    window.addEventListener('keydown', handler);
    return () => window.removeEventListener('keydown', handler);
  }, [next, prev, goTo]);

  return {
    scenarioIndex,
    simPhase,
    totalScenarios: TOTAL_SCENARIOS,
    totalSimPhases: SIM_SUB_PHASES,
    next,
    prev,
    goTo,
    isSimulation,
  };
}
