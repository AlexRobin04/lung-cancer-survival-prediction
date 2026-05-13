/** Kaplan–Meier 阶梯图（Recharts）与 Log-rank p 展示用工具 */

export const KM_GROUP_COLORS = ['#1565c0', '#c62828']

/** 与预测页「六基线」顺序一致，用于六模型最佳任务同图 KM 配色 */
export const KM_SIX_MODEL_ORDER = ['AMIL', 'DSMIL', 'EnsembleDecision', 'RRTMIL', 'S4MIL', 'WiKG']

export const KM_SIX_MODEL_COLORS = {
  AMIL: '#2e7d32',
  DSMIL: '#6a1b9a',
  EnsembleDecision: '#c62828',
  RRTMIL: '#1565c0',
  S4MIL: '#00838f',
  WiKG: '#ef6c00',
}

export const kmStrokeForModel = (modelType) =>
  KM_SIX_MODEL_COLORS[String(modelType || '').trim()] || '#455a64'

export const fmtFixed = (v, digits = 4) => {
  if (v === null || typeof v === 'undefined' || v === '') return '—'
  const n = Number(v)
  return Number.isFinite(n) ? n.toFixed(digits) : '—'
}

export const fmtLogRankP = (p) => {
  if (p === null || typeof p === 'undefined' || p === '') return '—'
  const n = Number(p)
  if (!Number.isFinite(n)) return '—'
  if (n <= 0) return String(n)
  if (n < 1e-6) return n.toExponential(3)
  if (n < 0.0001) return n.toFixed(6)
  return n.toFixed(4)
}

export const kmCurveKey = (label) => `S__${String(label || 'group').replace(/[^a-zA-Z0-9]+/g, '_')}`

export const kmSurvivalAt = (times, survival, t) => {
  const tt = times || []
  const ss = survival || []
  if (!tt.length) return 1
  let s = 1
  for (let i = 0; i < tt.length; i += 1) {
    if (Number(tt[i]) <= t) s = Number(ss[i])
    else break
  }
  return s
}

export const buildKmChartRows = (curves) => {
  const usable = (Array.isArray(curves) ? curves : []).filter((c) => Array.isArray(c?.times) && c.times.length > 0)
  if (usable.length === 0) return []
  const uniq = new Set([0])
  for (const c of usable) {
    for (const x of c.times || []) uniq.add(Number(x))
  }
  const ts = [...uniq].sort((a, b) => a - b)
  return ts.map((t) => {
    const row = { time: t }
    for (const c of usable) {
      row[kmCurveKey(c.label)] = kmSurvivalAt(c.times, c.survival, t)
    }
    return row
  })
}
