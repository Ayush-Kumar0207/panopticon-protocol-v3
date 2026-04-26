const BASE = '/api';

export async function resetEnvironment(taskLevel = 'easy') {
  const res = await fetch(`${BASE}/reset`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ task_level: taskLevel }),
  });
  if (!res.ok) throw new Error(await res.text());
  return res.json();
}

export async function stepEnvironment(actionType, target = '', subAction = 'none', reason = '') {
  const res = await fetch(`${BASE}/step`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ action_type: actionType, target, sub_action: subAction, reason }),
  });
  if (!res.ok) throw new Error(await res.text());
  return res.json();
}

export async function getObservation() {
  const res = await fetch(`${BASE}/observation`);
  if (!res.ok) throw new Error(await res.text());
  return res.json();
}

export async function getMetadata() {
  const res = await fetch(`${BASE}/metadata`);
  if (!res.ok) throw new Error(await res.text());
  return res.json();
}

export async function getHealth() {
  const res = await fetch(`${BASE}/health`);
  if (!res.ok) throw new Error(await res.text());
  return res.json();
}

export async function getAgentStatus() {
  const res = await fetch(`${BASE}/agent/status`);
  if (!res.ok) throw new Error(await res.text());
  return res.json();
}

export async function stepWithAgent() {
  const res = await fetch(`${BASE}/agent/step`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
  });
  if (!res.ok) throw new Error(await res.text());
  return res.json();
}
