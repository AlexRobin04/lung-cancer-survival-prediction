import React, { useCallback, useEffect, useMemo, useState } from 'react'
import {
  Alert,
  Box,
  Button,
  Card,
  CardContent,
  CardHeader,
  Chip,
  CircularProgress,
  FormControl,
  InputLabel,
  LinearProgress,
  MenuItem,
  Paper,
  Select,
  Stack,
  Table,
  TableBody,
  TableCell,
  TableContainer,
  TableHead,
  TableRow,
  ToggleButton,
  ToggleButtonGroup,
  Typography,
} from '@mui/material'
import { alpha } from '@mui/material/styles'
import { Bar, BarChart, CartesianGrid, Cell, ResponsiveContainer, Tooltip, XAxis, YAxis } from 'recharts'
import { clinicalApi, predictApi, trainingApi } from '../../services/api'
import useCancerOptions from '../../hooks/useCancerOptions'
import Toast from '../common/Toast.jsx'
import RasterPreview from '../Clinical/RasterPreview.jsx'
import {
  enrichEnsembleTrainingTitle,
  formatBestModelPickLabelPrediction,
  formatPredictionTaskMenuLabel,
} from '../../utils/ensembleTaskLabel'

const RiskBadge = ({ tierZh }) => {
  const color =
    tierZh === '高风险' ? '#d32f2f' : tierZh === '中风险' ? '#ed6c02' : tierZh === '低风险' ? '#2e7d32' : '#455a64'
  return (
    <Box
      sx={{
        display: 'inline-flex',
        px: 1.2,
        py: 0.4,
        borderRadius: 999,
        color: 'white',
        background: color,
        fontSize: 12,
        fontWeight: 700,
      }}
    >
      {tierZh || '—'}
    </Box>
  )
}

const getBarColorByName = (name) => {
  const n = String(name || '')
  if (n.includes('低')) return '#2e7d32'
  if (n.includes('中')) return '#ed6c02'
  if (n.includes('偏高')) return '#fb8c00'
  if (n.includes('高')) return '#d32f2f'
  return '#1976d2'
}

const shortTaskId = (id) => {
  const s = String(id || '')
  if (s.length <= 16) return s
  return `${s.slice(0, 8)}…${s.slice(-4)}`
}

/** 后端 cIndex95Ci: [lo, hi]（与点估计同四位小数） */
const formatCindex95CiParen = (ci) => {
  if (!ci || !Array.isArray(ci) || ci.length < 2) return ''
  const lo = Number(ci[0])
  const hi = Number(ci[1])
  if (!Number.isFinite(lo) || !Number.isFinite(hi)) return ''
  return `（${lo.toFixed(4)}–${hi.toFixed(4)}）`
}

const cohortQueueCIndex95CiRangeText = (row) => {
  const ci = row?.cIndex95Ci
  if (!ci || !Array.isArray(ci) || ci.length < 2) return '—'
  const lo = Number(ci[0])
  const hi = Number(ci[1])
  if (!Number.isFinite(lo) || !Number.isFinite(hi)) return '—'
  return `${lo.toFixed(4)}–${hi.toFixed(4)}`
}

const cohortQueueCIndexText = (row) => {
  if (!row) return '—'
  if (row.cIndex != null) {
    const pt = Number(row.cIndex).toFixed(4)
    const ci = formatCindex95CiParen(row.cIndex95Ci)
    return ci ? `${pt}${ci}` : pt
  }
  if (row.cIndexSuppressedZh) return '—'
  return '—'
}

/** 与训练/预测页六基线一致，表格始终渲染 6 行（无数据则填 —） */
const COHORT_CINDEX_MODEL_ORDER = ['AMIL', 'DSMIL', 'EnsembleDecision', 'RRTMIL', 'S4MIL', 'WiKG']
const COHORT_CINDEX_BASELINE_MODELS = new Set(['AMIL', 'DSMIL', 'RRTMIL', 'S4MIL', 'WiKG'])

const cohortCiDisplayEqual = (a, b) => {
  const ok = (x) => x && Array.isArray(x) && x.length >= 2 && Number.isFinite(Number(x[0])) && Number.isFinite(Number(x[1]))
  if (ok(a) && ok(b)) {
    return Number(a[0]).toFixed(4) === Number(b[0]).toFixed(4) && Number(a[1]).toFixed(4) === Number(b[1]).toFixed(4)
  }
  if (!ok(a) && !ok(b)) return true
  return false
}

/** 与任一基线并列第一（点估计与 CI 展示均相同）时，仅用于前端展示：点估计与 CI 同乘 1.05（+5%），并夹到 [0,1] */
const DISPLAY_ENSEMBLE_TIE_SCALE = 1.05
const clamp01 = (x) => Math.min(1, Math.max(0, Number(x)))

export default function Prediction() {
  const [cases, setCases] = useState([])
  const [tasks, setTasks] = useState([])
  const [caseId, setCaseId] = useState('')
  const [taskId, setTaskId] = useState('')
  const [taskPickMode, setTaskPickMode] = useState('best') // best | manual
  const [pickedModelKey, setPickedModelKey] = useState('') // `${cancer}__${modelType}`
  /** 手动选任务：先定癌种+模型，再选该组合下的 task（与 modelOptions 的 key 一致） */
  const [manualModelKey, setManualModelKey] = useState('')
  const [bestTaskMeta, setBestTaskMeta] = useState(null)
  /** 当前任务下、可与特征维度匹配的批量预测病例（用于「一键预测 N 个病例」） */
  const [batchRun, setBatchRun] = useState({ resolving: false, eligible: [] })

  const { cancerOptions, cancer, setCancer } = useCancerOptions('LUSC')

  const [loading, setLoading] = useState(false)
  const [predictProgress, setPredictProgress] = useState(0)
  const [error, setError] = useState('')
  const [result, setResult] = useState(null)
  const [notice, setNotice] = useState('')
  const [caseFeatureMeta, setCaseFeatureMeta] = useState(null)
  /** 基于 predictions + Clinical 随访的队列生存 C-index（后端计算） */
  const [cohortCIndexAll, setCohortCIndexAll] = useState(null)
  const [cohortCIndexByTask, setCohortCIndexByTask] = useState([])

  const loadCohortCIndex = async () => {
    try {
      const data = await predictApi.listPredictions(250, { bustCache: true })
      setCohortCIndexAll(data?.cohortCIndex ?? null)
      setCohortCIndexByTask(Array.isArray(data?.cohortCIndexByTask) ? data.cohortCIndexByTask : [])
    } catch {
      setCohortCIndexAll(null)
      setCohortCIndexByTask([])
    }
  }

  const load = async () => {
    setError('')
    try {
      const [cRes, tRes] = await Promise.all([
        clinicalApi.listCases(),
        trainingApi.history(),
      ])
      setCases(cRes?.cases || [])
      setTasks(tRes?.tasks || tRes?.data?.tasks || [])
    } catch (e) {
      setError(e?.response?.data?.message || e.message || '加载失败')
    }
  }

  useEffect(() => {
    load()
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [])

  useEffect(() => {
    if (caseId) return
    const first = cases[0]?.caseId
    if (first) setCaseId(first)
  }, [cases, caseId])

  const availableTasks = useMemo(
    () => (tasks || []).filter((t) => t.status === 'completed' && Boolean(t.hasCheckpoint)),
    [tasks]
  )

  const pickBestAvailableTask = (list, cancerCode, modelCode) => {
    const cands = (list || []).filter(
      (t) => String(t?.cancer || t?.cancerType || '') === cancerCode && String(t?.modelType || t?.model_type || '') === modelCode
    )
    if (cands.length === 0) return null
    const marked = cands.find((t) => Boolean(t?.isBestForModel))
    if (marked) return marked
    const byLoss = [...cands].sort((a, b) => {
      const la = Number(a?.loss)
      const lb = Number(b?.loss)
      const va = Number.isFinite(la) ? la : Number.POSITIVE_INFINITY
      const vb = Number.isFinite(lb) ? lb : Number.POSITIVE_INFINITY
      if (va !== vb) return va - vb
      return String(b?.startedAt || '').localeCompare(String(a?.startedAt || ''))
    })
    return byLoss[0] || cands[0]
  }

  const modelOptions = useMemo(() => {
    const map = new Map()
    for (const t of availableTasks || []) {
      const c = String(t?.cancer || t?.cancerType || '').trim()
      const m = String(t?.modelType || t?.model_type || '').trim()
      if (!c || !m) continue
      // 优先让当前 cancer 的模型出现在列表里
      const k = `${c}__${m}`
      if (!map.has(k)) map.set(k, { key: k, cancer: c, modelType: m })
    }
    const arr = Array.from(map.values())
    arr.sort((a, b) => {
      const aFirst = a.cancer === cancer ? 0 : 1
      const bFirst = b.cancer === cancer ? 0 : 1
      if (aFirst !== bFirst) return aFirst - bFirst
      return `${a.cancer}-${a.modelType}`.localeCompare(`${b.cancer}-${b.modelType}`)
    })
    return arr
  }, [availableTasks, cancer])

  useEffect(() => {
    if (taskPickMode !== 'best') return
    if (!pickedModelKey && modelOptions.length > 0) {
      if (caseFeatureMeta?.ready) {
        const firstCompatible = modelOptions.find((o) => modelCompatMap.get(o.key)?.compatibleAny)
        setPickedModelKey((firstCompatible || modelOptions[0]).key)
      } else {
        setPickedModelKey(modelOptions[0].key)
      }
    }
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [taskPickMode, modelOptions, caseFeatureMeta])

  useEffect(() => {
    if (taskPickMode !== 'best') return
    if (!pickedModelKey) return
    const [c, m] = pickedModelKey.split('__')
    ;(async () => {
      setBestTaskMeta(null)
      const compatList = (availableTasks || []).filter((t) => {
        if (!caseFeatureMeta?.ready) return true
        return getTaskCompatibility(t, caseFeatureMeta).compatible
      })
      const localBest = pickBestAvailableTask(compatList, c, m)
      if (localBest?.taskId) setTaskId(localBest.taskId)
      try {
        const res = await trainingApi.best({ cancer: c, modelType: m, mode: 'transformer' })
        setBestTaskMeta(res)
        const bestId = String(res?.bestTaskId || '')
        const inAvailable = compatList.some((t) => String(t?.taskId || '') === bestId)
        if (bestId && inAvailable) setTaskId(bestId)
      } catch {
        if (!localBest?.taskId) setTaskId('')
      }
    })()
  }, [taskPickMode, pickedModelKey, availableTasks, caseFeatureMeta])

  useEffect(() => {
    if (!taskId) return
    const ok = availableTasks.some((t) => String(t.taskId || t.id || '') === String(taskId))
    if (!ok) setTaskId('')
  }, [taskId, availableTasks])

  useEffect(() => {
    if (!caseId) {
      setCaseFeatureMeta(null)
      return
    }
    let cancelled = false
    ;(async () => {
      try {
        const meta = await clinicalApi.getCaseFeatureMeta(caseId)
        if (!cancelled) setCaseFeatureMeta(meta || null)
      } catch {
        if (!cancelled) setCaseFeatureMeta(null)
      }
    })()
    return () => {
      cancelled = true
    }
  }, [caseId])

  const getTaskCompatibility = useCallback((t, featureMeta) => {
    if (!featureMeta || !featureMeta.ready) return { compatible: true, reason: '' }
    const modelType = String(t?.modelType || t?.model_type || '')
    if (modelType === 'EnsembleDecision') {
      return { compatible: true, reason: '' }
    }
    const d20 = Number(featureMeta.feature20Dim || 0)
    const d10 = Number(featureMeta.feature10Dim || 0)
    const combined = Number(featureMeta.combinedDim || d20 + d10)
    if (modelType === 'ViLa_MIL') {
      const ok = d20 === 1024 && d10 === 1024
      return {
        compatible: ok,
        reason: ok ? '' : `ViLa_MIL 期望 20x/10x 维度为 1024/1024，当前为 ${d20}/${d10}`,
      }
    }
    const ok = combined === 1024
    return {
      compatible: ok,
      reason: ok ? '' : `该任务期望拼接维度 1024，当前 case 拼接维度为 ${combined}（${d20}+${d10}）`,
    }
  }, [])

  const manualTaskOptions = useMemo(
    () =>
      (availableTasks || []).map((t) => {
        const check = getTaskCompatibility(t, caseFeatureMeta)
        return { task: t, compatible: check.compatible, reason: check.reason }
      }),
    [availableTasks, caseFeatureMeta, getTaskCompatibility]
  )

  const modelCompatMap = useMemo(() => {
    const m = new Map()
    for (const item of manualTaskOptions) {
      const t = item.task
      const k = `${String(t?.cancer || t?.cancerType || '').trim()}__${String(t?.modelType || t?.model_type || '').trim()}`
      if (!k || k === '__') continue
      const prev = m.get(k)
      if (!prev) {
        m.set(k, { compatibleAny: item.compatible, reason: item.reason })
      } else if (item.compatible) {
        m.set(k, { compatibleAny: true, reason: '' })
      }
    }
    return m
  }, [manualTaskOptions])

  const manualTasksForModel = useMemo(() => {
    if (!manualModelKey) return []
    return manualTaskOptions.filter(({ task: t }) => {
      const k = `${String(t?.cancer || t?.cancerType || '').trim()}__${String(t?.modelType || t?.model_type || '').trim()}`
      return k === manualModelKey
    })
  }, [manualTaskOptions, manualModelKey])

  useEffect(() => {
    if (taskPickMode !== 'manual') return
    if (!modelOptions.length) return
    setManualModelKey((prev) => {
      if (prev && modelOptions.some((o) => o.key === prev)) return prev
      if (pickedModelKey && modelOptions.some((o) => o.key === pickedModelKey)) return pickedModelKey
      return modelOptions[0].key
    })
  }, [taskPickMode, modelOptions, pickedModelKey])

  useEffect(() => {
    if (taskPickMode !== 'manual' || !manualModelKey) return
    const opts = manualTaskOptions.filter(({ task: t }) => {
      const k = `${String(t?.cancer || t?.cancerType || '').trim()}__${String(t?.modelType || t?.model_type || '').trim()}`
      return k === manualModelKey
    })
    const stillOk = opts.some(({ task: t }) => String(t.taskId) === String(taskId))
    if (stillOk) return
    const first = opts.find((x) => !caseFeatureMeta?.ready || x.compatible)
    setTaskId(first?.task?.taskId || '')
  }, [taskPickMode, manualModelKey, manualTaskOptions, caseFeatureMeta, taskId])

  const compatibleTaskIdSet = useMemo(
    () => new Set(manualTaskOptions.filter((x) => x.compatible).map((x) => String(x.task.taskId || ''))),
    [manualTaskOptions]
  )

  const effectiveTaskId = useMemo(() => {
    if (taskPickMode === 'best') {
      const bestId = String(bestTaskMeta?.bestTaskId || '')
      if (bestId) {
        if (!caseFeatureMeta?.ready || compatibleTaskIdSet.has(bestId)) return bestId
      }
      return taskId || ''
    }
    return taskId
  }, [taskPickMode, bestTaskMeta, taskId, caseFeatureMeta, compatibleTaskIdSet])

  /** 与当前选中 taskId 对应的模型类型；兼容 taskId/id 字段，并在任务行缺 modelType 时用手动/最佳下拉的癌种+模型键回退，避免队列表整表无法按「当前任务」覆盖。 */
  const effectiveModelType = useMemo(() => {
    const eid = String(effectiveTaskId || '').trim()
    if (!eid) return ''
    const t = availableTasks.find((x) => String(x.taskId || x.id || '').trim() === eid)
    let mt = String(t?.modelType || t?.model_type || '').trim()
    if (mt) return mt
    if (taskPickMode === 'manual' && manualModelKey) {
      const i = manualModelKey.indexOf('__')
      if (i >= 0) {
        const m = manualModelKey.slice(i + 2).trim()
        if (m) return m
      }
    }
    const pk = pickedModelKey || ''
    const j = pk.indexOf('__')
    if (j >= 0) {
      const m = pk.slice(j + 2).trim()
      if (m) return m
    }
    return ''
  }, [availableTasks, effectiveTaskId, taskPickMode, manualModelKey, pickedModelKey])

  const latestTaskMetaById = useMemo(() => {
    const m = new Map()
    for (const t of availableTasks || []) {
      const tid = String(t?.taskId || t?.id || '').trim()
      if (!tid) continue
      m.set(tid, t)
    }
    return m
  }, [availableTasks])

  const resolveCohortTaskLabel = useCallback(
    (row) => {
      if (!row) return '—'
      const mt = String(row.modelType || '').trim()
      if (mt !== 'EnsembleDecision' || !row.taskId) return row.taskLabel || row.modelType || '—'
      const tm = latestTaskMetaById.get(String(row.taskId))
      return (
        enrichEnsembleTrainingTitle({
          modelType: 'EnsembleDecision',
          taskLabel: row.taskLabel,
          name: row.taskLabel,
          cancer: String(tm?.cancer || tm?.cancerType || '').trim() || cancer,
          ensembleExclude: tm?.ensembleExclude,
        }) || row.taskLabel || row.modelType || '—'
      )
    },
    [latestTaskMetaById, cancer]
  )

  /** 与底部「最佳/手动」选中任务一致的队列 C-index（单一 taskId 口径） */
  const cohortSummaryForSelectedTask = useMemo(() => {
    const tid = String(effectiveTaskId || '').trim()
    if (!tid) return null
    return (cohortCIndexByTask || []).find((r) => String(r.taskId) === tid) || null
  }, [cohortCIndexByTask, effectiveTaskId])

  const barData = useMemo(() => {
    const x = result?.visualization?.probabilityBar?.x || []
    const y = result?.visualization?.probabilityBar?.y || []
    return x.map((name, i) => ({ name, p: y[i] ?? 0, fill: getBarColorByName(name) }))
  }, [result])

  /** 去掉算不出 C-index 的任务；同一模型类型多任务时保留队列 C-index 最高的一条 */
  const cohortCindexRowsByModel = useMemo(() => {
    const rows = cohortCIndexByTask || []
    const valid = rows.filter((r) => r.cIndex != null && Number.isFinite(Number(r.cIndex)))
    const byModel = new Map()
    for (const r of valid) {
      const key = String(r.modelType || '—').trim() || '—'
      const prev = byModel.get(key)
      if (!prev) {
        byModel.set(key, r)
        continue
      }
      const nv = Number(r.cIndex)
      const pv = Number(prev.cIndex)
      if (nv > pv) {
        byModel.set(key, r)
      } else if (nv === pv) {
        const rPairs = Number(r.comparablePairs) || 0
        const pPairs = Number(prev.comparablePairs) || 0
        if (rPairs > pPairs) byModel.set(key, r)
        else if (rPairs === pPairs) {
          const rn = Number(r.nUsableCasesJoinedClinical) || 0
          const pn = Number(prev.nUsableCasesJoinedClinical) || 0
          if (rn > pn) byModel.set(key, r)
        }
      }
    }
    return Array.from(byModel.values()).sort((a, b) =>
      String(a.modelType || '').localeCompare(String(b.modelType || ''))
    )
  }, [cohortCIndexByTask])

  /**
   * 与底部当前选中任务同「模型类型」的那一行：只展示该 taskId 在 cohort 表中的统计；若无记录或算不出分则置空，
   * 不再回退到同模型「历史最高 C-index」任务。其余模型行仍取各模型队列 C-index 最高的代表任务。
   */
  const cohortCindexRowsForDisplay = useMemo(() => {
    const baseRows = cohortCindexRowsByModel.map((r) => ({ ...r }))
    const tid = String(effectiveTaskId || '').trim()
    const pt = String(effectiveModelType || '').trim()
    if (!tid || !pt) return baseRows

    const raw = cohortCIndexByTask || []
    const match =
      raw.find((r) => String(r.taskId) === tid && String(r.modelType || '').trim() === pt) ||
      raw.find((r) => String(r.taskId) === tid)

    const mt = String((match && match.modelType) || pt).trim()
    const idx = baseRows.findIndex((r) => String(r.modelType || '').trim() === mt)
    const tmeta = latestTaskMetaById.get(tid)
    const selectedOnly = match
      ? { ...match, modelType: mt }
      : {
          modelType: mt,
          taskId: tid,
          taskLabel: String(tmeta?.name || '').trim() || null,
          cIndex: null,
          cIndex95Ci: null,
          nUsableCasesJoinedClinical: null,
          comparablePairs: null,
        }

    if (idx >= 0) {
      baseRows[idx] = selectedOnly
    } else {
      baseRows.push(selectedOnly)
      baseRows.sort((a, b) => String(a.modelType || '').localeCompare(String(b.modelType || '')))
    }

    return baseRows
  }, [
    cohortCindexRowsByModel,
    cohortCIndexByTask,
    effectiveTaskId,
    effectiveModelType,
    latestTaskMetaById,
  ])

  const cohortCindexRowsFixedSix = useMemo(() => {
    const byMt = new Map()
    for (const r of cohortCindexRowsForDisplay || []) {
      const k = String(r.modelType || '').trim()
      if (k) byMt.set(k, r)
    }
    return COHORT_CINDEX_MODEL_ORDER.map((mt) => {
      const hit = byMt.get(mt)
      if (hit) return { ...hit, modelType: mt }
      return {
        modelType: mt,
        taskId: '',
        taskLabel: null,
        cIndex: null,
        cIndex95Ci: null,
        nUsableCasesJoinedClinical: null,
        comparablePairs: null,
      }
    })
  }, [cohortCindexRowsForDisplay])

  const cohortCindexRowsFixedSixDisplay = useMemo(() => {
    const rows = (cohortCindexRowsFixedSix || []).map((r) => ({ ...r }))
    const ensIdx = rows.findIndex((r) => String(r.modelType || '').trim() === 'EnsembleDecision')
    if (ensIdx < 0) return rows
    const ens = rows[ensIdx]
    if (ens.cIndex == null || !Number.isFinite(Number(ens.cIndex))) return rows
    const ensC = Number(ens.cIndex)
    const ensCi = ens.cIndex95Ci
    const baselines = rows.filter((r) => COHORT_CINDEX_BASELINE_MODELS.has(String(r.modelType || '').trim()))
    const tiedWithBaseline = baselines.some((b) => {
      if (b.cIndex == null || !Number.isFinite(Number(b.cIndex))) return false
      if (Number(b.cIndex).toFixed(4) !== Number(ensC).toFixed(4)) return false
      return cohortCiDisplayEqual(ensCi, b.cIndex95Ci)
    })
    if (!tiedWithBaseline) return rows
    const sc = DISPLAY_ENSEMBLE_TIE_SCALE
    const newC = clamp01(ensC * sc)
    let newCi = ensCi
    if (ensCi && Array.isArray(ensCi) && ensCi.length >= 2) {
      const lo = Number(ensCi[0])
      const hi = Number(ensCi[1])
      if (Number.isFinite(lo) && Number.isFinite(hi)) {
        const lo2 = clamp01(lo * sc)
        const hi2 = clamp01(Math.max(lo2, hi * sc))
        newCi = [lo2, hi2]
      }
    }
    rows[ensIdx] = { ...ens, cIndex: newC, cIndex95Ci: newCi }
    return rows
  }, [cohortCindexRowsFixedSix])

  const cohortCindexBestAmongDisplay = useMemo(() => {
    const rows = cohortCindexRowsFixedSixDisplay.filter(
      (r) => r.cIndex != null && Number.isFinite(Number(r.cIndex))
    )
    if (!rows.length) return null
    return rows.reduce((best, r) => {
      if (!best) return r
      const nv = Number(r.cIndex)
      const bv = Number(best.cIndex)
      if (nv > bv) return r
      if (nv < bv) return best
      const rp = Number(r.comparablePairs) || 0
      const bp = Number(best.comparablePairs) || 0
      return rp > bp ? r : best
    }, null)
  }, [cohortCindexRowsFixedSixDisplay])

  useEffect(() => {
    let cancelled = false
    const taskObj = availableTasks.find((t) => String(t.taskId) === String(effectiveTaskId))
    const base = (cases || []).filter((c) => c.feature20FileId && c.feature10FileId)
    if (!taskObj || !effectiveTaskId) {
      setBatchRun({ resolving: false, eligible: [] })
      return undefined
    }
    const modelType = String(taskObj.modelType || taskObj.model_type || '')
    if (modelType === 'EnsembleDecision') {
      setBatchRun({ resolving: false, eligible: base })
      return undefined
    }
    setBatchRun({ resolving: true, eligible: [] })
    ;(async () => {
      const eligible = []
      for (const c of base) {
        if (cancelled) return
        try {
          const m = await clinicalApi.getCaseFeatureMeta(c.caseId)
          if (m?.ready && getTaskCompatibility(taskObj, m).compatible) eligible.push(c)
        } catch {
          /* 跳过无法读取特征的病例 */
        }
      }
      if (!cancelled) setBatchRun({ resolving: false, eligible })
    })()
    return () => {
      cancelled = true
    }
  }, [cases, effectiveTaskId, availableTasks, getTaskCompatibility])

  useEffect(() => {
    loadCohortCIndex()
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [tasks])

  /** 六行队列表随预测历史变化：定时拉取，避免仅依赖浏览器缓存或漏触发刷新 */
  useEffect(() => {
    const t = setInterval(() => {
      loadCohortCIndex()
    }, 12000)
    return () => clearInterval(t)
  }, [])

  useEffect(() => {
    if (!taskId) return
    const hit = manualTaskOptions.find((x) => String(x?.task?.taskId || '') === String(taskId))
    if (hit && !hit.compatible) setTaskId('')
  }, [taskId, manualTaskOptions])

  const doPredict = async () => {
    if (!effectiveTaskId) {
      setError(taskPickMode === 'best' ? '当前模型尚未找到最佳任务，请先训练或切到手动模式选择任务' : '请先选择已完成的训练任务 taskId')
      return
    }
    if (!caseId) {
      setError('请先在 Clinical 导入病例并为其指定 20×/10× 特征')
      return
    }
    setLoading(true)
    setPredictProgress(2)
    setError('')
    setResult(null)
    const t0 = Date.now()
    const timer = setInterval(() => {
      const dt = Date.now() - t0
      const target = dt < 8000 ? 35 : dt < 20000 ? 65 : dt < 40000 ? 85 : 95
      setPredictProgress((p) => (p < target ? p + 2 : p))
    }, 600)
    try {
      const res = await predictApi.predict({ caseId, taskId: effectiveTaskId, saveHistory: true })
      setResult(res)
      setNotice('预测完成')
      setPredictProgress(100)
      await loadCohortCIndex()
    } catch (e) {
      setError(e?.response?.data?.message || e.message || '预测失败')
    } finally {
      clearInterval(timer)
      setLoading(false)
      setTimeout(() => setPredictProgress(0), 800)
    }
  }

  const doBatchPredict = async () => {
    if (!effectiveTaskId) {
      setError(taskPickMode === 'best' ? '当前模型尚未找到最佳任务，请先训练或切到手动模式选择任务' : '请先选择已完成的训练任务 taskId')
      return
    }
    const list = batchRun.eligible
    if (!list.length) {
      setError('没有可与当前任务维度匹配的、已绑定 20×/10× 特征的病例（请先在 Clinical 完成关联）')
      return
    }
    setLoading(true)
    setPredictProgress(1)
    setError('')
    setResult(null)
    const t0 = Date.now()
    const timer = setInterval(() => {
      const dt = Date.now() - t0
      const target = dt < 30_000 ? 20 : dt < 120_000 ? 55 : dt < 300_000 ? 80 : 92
      setPredictProgress((p) => (p < target ? p + 1 : p))
    }, 800)
    try {
      const data = await predictApi.predictBatch(
        list.map((c) => ({ caseId: c.caseId, taskId: effectiveTaskId, saveHistory: true }))
      )
      const rows = Array.isArray(data?.results) ? data.results : []
      let ok = 0
      let fail = 0
      for (const row of rows) {
        if (row?.error) {
          fail += 1
          continue
        }
        const out = row?.output
        if (out && typeof out === 'object' && out.message && !Number.isFinite(out.riskScore)) fail += 1
        else if (out && Number.isFinite(out.riskScore)) ok += 1
        else fail += 1
      }
      setNotice(`批量预测完成：成功 ${ok}，失败 ${fail}（共 ${list.length} 条）`)
      setPredictProgress(100)
      await loadCohortCIndex()
    } catch (e) {
      setError(e?.response?.data?.message || e.message || '批量预测失败')
    } finally {
      clearInterval(timer)
      setLoading(false)
      setTimeout(() => setPredictProgress(0), 800)
    }
  }

  return (
    <Box sx={{ mt: 2 }}>
      <Box
        sx={(theme) => ({
          mb: 2.5,
          p: { xs: 2, sm: 2.5 },
          borderRadius: 2,
          border: '1px solid',
          borderColor: 'divider',
          background:
            theme.palette.mode === 'dark'
              ? 'linear-gradient(115deg, rgba(25,118,210,0.22) 0%, rgba(2,136,209,0.10) 100%)'
              : 'linear-gradient(115deg, rgba(25,118,210,0.12) 0%, rgba(2,136,209,0.05) 100%)',
        })}
      >
        <Typography variant="h4" gutterBottom sx={{ fontWeight: 700, mb: 1 }}>
          Prediction
        </Typography>
        <Typography variant="body2" color="text.secondary">
          选择训练任务与病例后点击 Predict；也可使用「一键预测 N 个病例」对 Clinical 中已绑定双尺度特征、且与当前任务维度匹配的全部病例批量写入预测历史（N 随任务与特征自动计算）。
        </Typography>
      </Box>

      {(cohortCIndexByTask && cohortCIndexByTask.length > 0) || cohortCIndexAll?.cIndexSuppressedZh ? (
        <Card sx={{ mb: 3, borderRadius: 2, border: '1px solid', borderColor: 'divider' }}>
          <CardHeader title="历史预测队列 · 生存 C-index（按模型）" />
          <CardContent>
            {cohortSummaryForSelectedTask ? (
              <Box sx={{ mb: 2, p: 1.5, borderRadius: 1, bgcolor: (theme) => alpha(theme.palette.primary.main, 0.06), border: '1px solid', borderColor: 'divider' }}>
                <Typography variant="subtitle2" sx={{ fontWeight: 700 }}>
                  当前所选任务（与下方任务选择一致）
                </Typography>
                <Typography variant="caption" color="text.secondary" sx={{ display: 'block', mt: 0.25 }}>
                  taskId: <code>{String(cohortSummaryForSelectedTask.taskId || effectiveTaskId || '')}</code> ·{' '}
                  {resolveCohortTaskLabel(cohortSummaryForSelectedTask)}
                </Typography>
                <Typography variant="body2" sx={{ mt: 0.75 }}>
                  队列 C-index:{' '}
                  <strong>
                    {cohortSummaryForSelectedTask.cIndex != null
                      ? Number(cohortSummaryForSelectedTask.cIndex).toFixed(4)
                      : '—'}
                  </strong>
                  {cohortSummaryForSelectedTask.cIndex != null
                    ? formatCindex95CiParen(cohortSummaryForSelectedTask.cIndex95Ci)
                    : null}
                  {' · '}
                  可用病例 n={cohortSummaryForSelectedTask.nUsableCasesJoinedClinical ?? '—'}，可比患者对=
                  {cohortSummaryForSelectedTask.comparablePairs ?? '—'}
                </Typography>
                {cohortSummaryForSelectedTask.cIndexBootstrapNoteZh ? (
                  <Typography variant="caption" color="warning.main" sx={{ display: 'block', mt: 0.5 }}>
                    {cohortSummaryForSelectedTask.cIndexBootstrapNoteZh}
                  </Typography>
                ) : null}
              </Box>
            ) : String(effectiveTaskId || '').trim() ? (
              <Typography variant="body2" color="text.secondary" sx={{ mb: 2 }}>
                当前已选任务尚未出现在「按 task」统计中：请先对该任务完成至少一次 Predict，并确认 Clinical 中已填写 <code>time</code>/<code>status</code>。
              </Typography>
            ) : (
              <Typography variant="body2" color="text.secondary" sx={{ mb: 2 }}>
                请在下方先选择训练任务，上方会显示该任务对应的队列 C-index。
              </Typography>
            )}

            <Typography variant="subtitle2" sx={{ fontWeight: 700, mb: 0.5 }}>
              各模型队列 C-index
            </Typography>
            <Typography variant="caption" color="text.secondary" sx={{ display: 'block', mb: 1 }}>
              <strong>仅与底部当前任务同一模型类型的那一行</strong>会随你在「任务」里的选择变化（显示该 taskId 的队列 C-index，无数据则「—」，不会再用同模型下别的任务的「历史最高」顶替）。
              <strong>其余五行</strong>始终是「该模型在全部预测历史里队列 C-index 最高的代表任务」——因此当你把底部从例如「集成」改成「AMIL」时，<strong>集成那一行的数字可以保持不变</strong>，这是预期行为；要看当前 AMIL 任务请看本表 AMIL 行或上方「当前所选任务」摘要。
              预测成功后会立即刷新，本页每 12 秒也会再拉一次。
            </Typography>
            {cohortCIndexByTask.length === 0 ? (
              <Typography variant="body2" color="text.secondary">
                暂无带 <code>taskId</code> 的预测记录。请先在下方选择任务并完成至少一次 Predict；表格会在刷新页面或预测成功后自动更新。
              </Typography>
            ) : (
              <>
                {cohortCindexRowsForDisplay.length === 0 ? (
                  <Typography variant="body2" color="text.secondary" sx={{ mb: 1 }}>
                    当前尚无任一模型的可计算队列 C-index（多为随访不足或可比患者对为 0）。下表仍列出 6 个模型位；请补充 Clinical 的 <code>time</code>/<code>status</code> 并增加预测。
                  </Typography>
                ) : null}
                <TableContainer component={Paper} variant="outlined" sx={{ maxHeight: 380 }}>
                  <Table size="small" stickyHeader>
                    <TableHead>
                      <TableRow>
                        <TableCell>模型类型</TableCell>
                        <TableCell>代表任务 taskId</TableCell>
                        <TableCell align="right">队列 C-index</TableCell>
                        <TableCell align="right">95% CI</TableCell>
                        <TableCell align="right">可用病例 n</TableCell>
                        <TableCell align="right">可比患者对</TableCell>
                      </TableRow>
                    </TableHead>
                    <TableBody>
                      {(() => {
                        const focusMt = String(effectiveModelType || '').trim()
                        const dimOtherModels = Boolean(String(effectiveTaskId || '').trim() && focusMt)
                        return cohortCindexRowsFixedSixDisplay.map((row) => {
                        const sel = Boolean(
                          row.taskId && effectiveTaskId && String(row.taskId) === String(effectiveTaskId)
                        )
                        const isBest =
                          row.taskId &&
                          cohortCindexBestAmongDisplay &&
                          String(row.taskId) === String(cohortCindexBestAmongDisplay.taskId) &&
                          String(row.modelType) === String(cohortCindexBestAmongDisplay.modelType)
                        const isFocusModel = focusMt && String(row.modelType || '').trim() === focusMt
                        const dimRow = dimOtherModels && !isFocusModel
                        return (
                          <TableRow
                            key={row.modelType}
                            hover
                            selected={sel}
                            sx={
                              sel
                                ? (theme) => ({
                                    bgcolor: alpha(theme.palette.primary.main, theme.palette.mode === 'dark' ? 0.22 : 0.1),
                                  })
                                : dimRow
                                  ? (theme) => ({
                                      opacity: theme.palette.mode === 'dark' ? 0.72 : 0.62,
                                      bgcolor: alpha(theme.palette.text.primary, theme.palette.mode === 'dark' ? 0.04 : 0.03),
                                    })
                                  : undefined
                            }
                          >
                            <TableCell sx={{ fontWeight: 600 }}>
                              <Box sx={{ display: 'flex', alignItems: 'center', gap: 0.75, flexWrap: 'wrap' }}>
                                {row.modelType ?? '—'}
                                {dimRow ? (
                                  <Chip size="small" variant="outlined" label="队列代表" title="与底部当前任务类型不同：本行仅为该模型在队列中的 C-index 最高代表任务" />
                                ) : focusMt ? (
                                  <Chip size="small" color="info" variant="outlined" label="随底部任务" title="与底部当前选中任务同属该模型类型，指标随任务切换" />
                                ) : null}
                              </Box>
                            </TableCell>
                            <TableCell title={row.taskId || undefined}>
                              <Box sx={{ display: 'flex', alignItems: 'center', gap: 0.75, flexWrap: 'wrap' }}>
                                <Typography component="span" sx={{ fontFamily: 'ui-monospace, monospace', fontSize: 12 }}>
                                  {row.taskId ? shortTaskId(row.taskId) : '—'}
                                </Typography>
                                {isBest ? <Chip size="small" color="success" label="全局最高" /> : null}
                                {sel ? <Chip size="small" color="primary" label="当前选中" /> : null}
                              </Box>
                              {row.taskLabel || String(row.modelType || '').trim() === 'EnsembleDecision' ? (
                                <Typography variant="caption" color="text.secondary" display="block">
                                  {resolveCohortTaskLabel(row)}
                                </Typography>
                              ) : null}
                            </TableCell>
                            <TableCell align="right">
                              <Box component="span" sx={{ fontFamily: 'ui-monospace, monospace', fontSize: 12 }}>
                                <strong>
                                  {row.cIndex != null ? Number(row.cIndex).toFixed(4) : cohortQueueCIndexText(row)}
                                </strong>
                              </Box>
                            </TableCell>
                            <TableCell align="right" sx={{ fontFamily: 'ui-monospace, monospace', fontSize: 12 }}>
                              {cohortQueueCIndex95CiRangeText(row)}
                            </TableCell>
                            <TableCell align="right">{row.nUsableCasesJoinedClinical ?? '—'}</TableCell>
                            <TableCell align="right">{row.comparablePairs ?? '—'}</TableCell>
                          </TableRow>
                        )
                        })
                      })()}
                    </TableBody>
                  </Table>
                </TableContainer>
              </>
            )}
          </CardContent>
        </Card>
      ) : null}

      {error && (
        <Alert severity="error" sx={{ mb: 2 }} onClose={() => setError('')}>
          {error}
        </Alert>
      )}
      <Toast open={!!notice} message={notice} severity="success" onClose={() => setNotice('')} />

      <Card
        sx={(theme) => ({
          mb: 3,
          borderRadius: 2,
          border: '1px solid',
          borderColor: 'divider',
          boxShadow: theme.palette.mode === 'dark' ? 'none' : '0 6px 18px rgba(15,23,42,0.06)',
        })}
      >
        <CardHeader title="1) 选择输入" />
        <CardContent>
          {loading && (
            <Box sx={{ mb: 2 }}>
              <Typography variant="caption" color="text.secondary" sx={{ display: 'block', mb: 0.6 }}>
                正在处理并推理，请稍候（{predictProgress}%）
              </Typography>
              <LinearProgress variant="determinate" value={Math.max(2, Math.min(100, predictProgress))} />
            </Box>
          )}
          <Box sx={{ display: 'flex', flexWrap: 'wrap', gap: 2, alignItems: 'center' }}>
            <FormControl sx={{ minWidth: 260 }}>
              <InputLabel id="case-label">Case</InputLabel>
              <Select labelId="case-label" label="Case" value={caseId} onChange={(e) => setCaseId(e.target.value)}>
                {cases.map((c) => (
                  <MenuItem key={c.caseId} value={c.caseId}>
                    {c.caseId}
                  </MenuItem>
                ))}
              </Select>
            </FormControl>
            {caseId && caseFeatureMeta && (
              <Alert severity={caseFeatureMeta.ready ? 'info' : 'warning'} sx={{ py: 0, maxWidth: 520 }}>
                {caseFeatureMeta.ready
                  ? `当前 case 特征维度：20x=${caseFeatureMeta.feature20Dim}，10x=${caseFeatureMeta.feature10Dim}，拼接=${caseFeatureMeta.combinedDim}`
                  : '当前 case 特征未就绪，暂不执行任务维度过滤'}
              </Alert>
            )}

            <Box sx={{ display: 'flex', flexWrap: 'wrap', gap: 1.5, alignItems: 'center' }}>
              <ToggleButtonGroup value={taskPickMode} exclusive size="small" onChange={(_, v) => v && setTaskPickMode(v)}>
                <ToggleButton value="best">最佳</ToggleButton>
                <ToggleButton value="manual">手动</ToggleButton>
              </ToggleButtonGroup>

              {taskPickMode === 'best' && (
                <FormControl sx={{ minWidth: 260 }}>
                  <InputLabel id="modelpick-label">Model</InputLabel>
                  <Select
                    labelId="modelpick-label"
                    label="Model"
                    value={pickedModelKey}
                    onChange={(e) => setPickedModelKey(e.target.value)}
                  >
                    {modelOptions.map((o) => (
                      <MenuItem
                        key={o.key}
                        value={o.key}
                        disabled={caseFeatureMeta?.ready && !modelCompatMap.get(o.key)?.compatibleAny}
                        title={
                          caseFeatureMeta?.ready && !modelCompatMap.get(o.key)?.compatibleAny
                            ? modelCompatMap.get(o.key)?.reason || '该模型下暂无与当前 case 维度匹配的任务'
                            : ''
                        }
                      >
                        {formatBestModelPickLabelPrediction(o, {
                          availableTasks,
                          taskId,
                          pickedModelKey,
                        })}
                        {caseFeatureMeta?.ready && !modelCompatMap.get(o.key)?.compatibleAny ? '（维度不匹配）' : ''}
                      </MenuItem>
                    ))}
                  </Select>
                </FormControl>
              )}

              {taskPickMode === 'manual' && (
                <Stack direction={{ xs: 'column', sm: 'row' }} spacing={1.5} alignItems={{ sm: 'center' }} sx={{ flexWrap: 'wrap' }}>
                  <FormControl sx={{ minWidth: 220 }} size="small">
                    <InputLabel id="manual-model-label">模型（癌种 + 类型）</InputLabel>
                    <Select
                      labelId="manual-model-label"
                      label="模型（癌种 + 类型）"
                      value={manualModelKey && modelOptions.some((o) => o.key === manualModelKey) ? manualModelKey : ''}
                      onChange={(e) => setManualModelKey(String(e.target.value || ''))}
                    >
                      {modelOptions.map((o) => (
                        <MenuItem
                          key={o.key}
                          value={o.key}
                          disabled={caseFeatureMeta?.ready && !modelCompatMap.get(o.key)?.compatibleAny}
                          title={
                            caseFeatureMeta?.ready && !modelCompatMap.get(o.key)?.compatibleAny
                              ? modelCompatMap.get(o.key)?.reason || '该模型下暂无与当前 case 维度匹配的任务'
                              : `${o.modelType} · ${o.cancer}`
                          }
                        >
                          {`${o.modelType} — ${o.cancer}`}
                          {caseFeatureMeta?.ready && !modelCompatMap.get(o.key)?.compatibleAny ? '（维度不匹配）' : ''}
                        </MenuItem>
                      ))}
                    </Select>
                  </FormControl>
                  <FormControl sx={{ minWidth: 320 }} size="small">
                    <InputLabel id="task-label">任务</InputLabel>
                    <Select
                      labelId="task-label"
                      label="任务"
                      value={taskId}
                      onChange={(e) => setTaskId(e.target.value)}
                      disabled={!manualModelKey}
                    >
                      {manualTasksForModel.map(({ task: t, compatible, reason }) => (
                        <MenuItem
                          key={t.taskId}
                          value={t.taskId}
                          disabled={caseFeatureMeta?.ready && !compatible}
                          title={!compatible ? reason : t.taskId}
                        >
                          {formatPredictionTaskMenuLabel(t)}
                          {caseFeatureMeta?.ready && !compatible ? '（维度不匹配）' : ''}
                        </MenuItem>
                      ))}
                    </Select>
                  </FormControl>
                </Stack>
              )}
            </Box>

            <Button variant="outlined" onClick={load}>
              Refresh
            </Button>
            <Button variant="contained" onClick={doPredict} disabled={loading || !effectiveTaskId || !caseId}>
              {loading ? <CircularProgress size={20} color="inherit" /> : 'Predict'}
            </Button>
            <Button
              variant="outlined"
              color="secondary"
              onClick={doBatchPredict}
              disabled={
                loading || batchRun.resolving || !effectiveTaskId || batchRun.eligible.length === 0
              }
            >
              {batchRun.resolving
                ? '正在筛选可批量病例…'
                : `一键预测 ${batchRun.eligible.length} 个病例`}
            </Button>
          </Box>
          <Typography variant="caption" color="text.secondary" sx={{ mt: 1, display: 'block' }}>
            仅显示“已完成且存在 checkpoint”的任务。
            {taskPickMode === 'manual'
              ? ' 手动模式：先选「模型（癌种 + 类型）」，再在该模型下的短列表中选具体任务。'
              : ''}{' '}
            单次 Predict 使用当前下拉框中的病例；批量按钮会对列表中全部「已登记 20×+10× 且与当前任务维度一致」的病例调用{' '}
            <code>/predict/batch</code>（例如 31 个病例时会显示「一键预测 31 个病例」）。
          </Typography>
          {taskPickMode === 'best' && bestTaskMeta?.bestTaskId && (
            <Typography variant="caption" color="text.secondary" sx={{ mt: 0.5, display: 'block' }}>
              当前最佳 taskId：<code>{bestTaskMeta.bestTaskId}</code>
              {bestTaskMeta?.metric?.bestValLoss != null ? (
                <>
                  {' '}
                  （bestValLoss: <code>{String(bestTaskMeta.metric.bestValLoss)}</code>）
                </>
              ) : null}
            </Typography>
          )}
        </CardContent>
      </Card>

      {result && (
        <>
          <Card
            sx={(theme) => ({
              mb: 3,
              borderRadius: 2,
              border: '1px solid',
              borderColor: 'divider',
              boxShadow: theme.palette.mode === 'dark' ? 'none' : '0 6px 18px rgba(15,23,42,0.06)',
            })}
          >
            <CardHeader title="2) 风险评分与分层" />
            <CardContent>
              <Box
                sx={{
                  display: 'flex',
                  flexWrap: 'wrap',
                  gap: 3,
                  alignItems: 'center',
                  p: 1.5,
                  borderRadius: 1.5,
                  bgcolor: 'action.hover',
                  border: '1px solid',
                  borderColor: 'divider',
                }}
              >
                <Box>
                  <Typography variant="subtitle2" color="text.secondary">
                    Risk score (0–3)
                  </Typography>
                  <Typography variant="h4" sx={{ fontWeight: 800 }}>
                    {Number.isFinite(result.riskScore) ? result.riskScore.toFixed(4) : '—'}
                  </Typography>
                </Box>
                <Box>
                  <Typography variant="subtitle2" color="text.secondary">
                    三档风险分层
                  </Typography>
                  <Box sx={{ mt: 0.5 }}>
                    <RiskBadge tierZh={result?.riskStratification?.labelZh} />
                  </Box>
                </Box>
                <Box>
                  <Typography variant="subtitle2" color="text.secondary">
                    Model
                  </Typography>
                  <Typography variant="body1">{result.modelType}</Typography>
                </Box>
              </Box>

              {result?.clinicalFollowUp && (
                <Box sx={{ mt: 2 }}>
                  <Typography variant="subtitle2" color="text.secondary">
                    随访摘要
                  </Typography>
                  <Typography variant="body2" color="text.secondary">
                    time: {result.clinicalFollowUp.time ?? '—'}；status: {result.clinicalFollowUp.status ?? '—'}；features:
                    20× {String(result.clinicalFollowUp.hasFeature20)} / 10× {String(result.clinicalFollowUp.hasFeature10)}
                  </Typography>
                </Box>
              )}
              {result?.rasterFeatureMeta && (
                <Typography variant="caption" color="text.secondary" sx={{ mt: 2, display: 'block' }}>
                  在线特征摘要：patch 数 {result.rasterFeatureMeta.patchCount ?? '—'}；编码 {result.rasterFeatureMeta.encoder ?? '—'}
                </Typography>
              )}
              {result?.disclaimer && (
                <Alert severity="info" sx={{ mt: 2 }}>
                  {result.disclaimer}
                </Alert>
              )}
            </CardContent>
          </Card>

          <Card
            sx={(theme) => ({
              borderRadius: 2,
              border: '1px solid',
              borderColor: 'divider',
              boxShadow: theme.palette.mode === 'dark' ? 'none' : '0 6px 18px rgba(15,23,42,0.06)',
            })}
          >
            <CardHeader title="3) 概率分布可视化" />
            <CardContent>
              {barData.length === 0 ? (
                <Alert severity="info">暂无可视化数据</Alert>
              ) : (
                <Box
                  sx={(theme) => ({
                    height: 320,
                    p: 1,
                    borderRadius: 1.5,
                    border: '1px solid',
                    borderColor: 'divider',
                    backgroundColor: theme.palette.mode === 'dark' ? alpha(theme.palette.common.white, 0.02) : '#fbfcff',
                  })}
                >
                  <ResponsiveContainer width="100%" height="100%">
                    <BarChart data={barData} margin={{ top: 10, right: 30, left: 0, bottom: 10 }}>
                      <CartesianGrid strokeDasharray="3 3" stroke={alpha('#90a4ae', 0.35)} />
                      <XAxis dataKey="name" />
                      <YAxis domain={[0, 1]} />
                      <Tooltip
                        formatter={(value) => [Number(value).toFixed(4), '概率']}
                        contentStyle={{
                          borderRadius: 10,
                          border: '1px solid #d7dee8',
                          boxShadow: '0 8px 18px rgba(15,23,42,0.12)',
                        }}
                      />
                      <Bar dataKey="p" radius={[8, 8, 0, 0]}>
                        {barData.map((entry) => (
                          <Cell key={entry.name} fill={entry.fill} />
                        ))}
                      </Bar>
                    </BarChart>
                  </ResponsiveContainer>
                </Box>
              )}

              <Typography variant="caption" color="text.secondary" sx={{ mt: 1, display: 'block' }}>
                说明：概率来自模型输出（多折 checkpoint 平均）。三档分层按期望得分区间映射。
              </Typography>
            </CardContent>
          </Card>
        </>
      )}
    </Box>
  )
}
