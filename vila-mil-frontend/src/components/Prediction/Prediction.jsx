import React, { useCallback, useEffect, useMemo, useState } from 'react'
import {
  Alert,
  Box,
  Button,
  Card,
  CardContent,
  CardHeader,
  Checkbox,
  Chip,
  CircularProgress,
  Collapse,
  Dialog,
  DialogActions,
  DialogContent,
  DialogTitle,
  FormControl,
  InputLabel,
  LinearProgress,
  List,
  ListItem,
  ListItemIcon,
  ListItemText,
  MenuItem,
  Paper,
  Select,
  Stack,
  Table,
  TextField,
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
import { Bar, BarChart, CartesianGrid, Cell, Legend, ResponsiveContainer, Tooltip, XAxis, YAxis } from 'recharts'
import { clinicalApi, predictApi, trainingApi } from '../../services/api'
import { readEnsembleTiebreakAllowFallback } from '../../constants/ensemblePredictPrefs'
import useCancerOptions from '../../hooks/useCancerOptions'
import { useCohortAndKmFromPredictions } from '../../hooks/useCohortAndKmFromPredictions'
import Toast from '../common/Toast.jsx'
import RasterPreview from '../Clinical/RasterPreview.jsx'
import {
  enrichEnsembleTrainingTitle,
  formatBestModelPickLabelPrediction,
  formatPredictionTaskMenuLabel,
} from '../../utils/ensembleTaskLabel'
import {
  cohortHazardRatioCellText,
  cohortQueueCIndex95CiRangeText,
  cohortQueueCIndexText,
  formatCindex95CiParen,
  shortTaskId,
} from '../../utils/predictionCohortKmUtils'

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

  const [batchCaseDialogOpen, setBatchCaseDialogOpen] = useState(false)
  const [batchSelectedCases, setBatchSelectedCases] = useState(new Set())
  const [batchRangeFrom, setBatchRangeFrom] = useState('1')
  const [batchRangeTo, setBatchRangeTo] = useState('1')

  const [historyExpanded, setHistoryExpanded] = useState(false)
  const [predictionRecords, setPredictionRecords] = useState([])
  const [selectedRecordIds, setSelectedRecordIds] = useState(new Set())
  const [predictionRecordsLoading, setPredictionRecordsLoading] = useState(false)

  const loadPredictionRecords = useCallback(async () => {
    setPredictionRecordsLoading(true)
    try {
      const res = await predictApi.listPredictions(200, { bustCache: true })
      setPredictionRecords(res?.items || [])
    } catch {
      // silent
    } finally {
      setPredictionRecordsLoading(false)
    }
  }, [])

  const handleSelectRecord = (id) => {
    setSelectedRecordIds((prev) => {
      const next = new Set(prev)
      if (next.has(id)) next.delete(id)
      else next.add(id)
      return next
    })
  }

  const handleSelectAllRecords = () => {
    setSelectedRecordIds((prev) => {
      if (prev.size === predictionRecords.length) return new Set()
      return new Set(predictionRecords.map((r) => r.id || '').filter(Boolean))
    })
  }

  const handleDeleteSelectedRecords = async () => {
    const ids = Array.from(selectedRecordIds).filter(Boolean)
    if (!ids.length) return
    if (!window.confirm(`确定删除选中的 ${ids.length} 条预测记录吗？`)) return
    try {
      await predictApi.deletePredictions({ ids })
      setNotice(`已删除 ${ids.length} 条记录`)
      setSelectedRecordIds(new Set())
      await loadPredictionRecords()
      await ck.reloadCohortAndKm()
    } catch (e) {
      setError(e?.response?.data?.message || e.message || '删除失败')
    }
  }

  const handleDeleteAllRecords = async () => {
    if (!window.confirm('确定删除全部预测历史记录吗？此操作不可恢复。')) return
    try {
      await predictApi.deletePredictions({ deleteAll: true })
      setNotice('已清空全部预测记录')
      setSelectedRecordIds(new Set())
      await loadPredictionRecords()
      await ck.reloadCohortAndKm()
    } catch (e) {
      setError(e?.response?.data?.message || e.message || '清空失败')
    }
  }

  const [loading, setLoading] = useState(false)
  const [predictProgress, setPredictProgress] = useState(0)
  const [error, setError] = useState('')
  const [result, setResult] = useState(null)
  const [notice, setNotice] = useState('')
  const [caseFeatureMeta, setCaseFeatureMeta] = useState(null)

  const load = async () => {
    setError('')
    try {
      const [cRes, tRes, pRes] = await Promise.all([
        clinicalApi.listCases(),
        trainingApi.history(),
        predictApi.listPredictions(200, { bustCache: true }).catch(() => null),
      ])
      setCases(cRes?.cases || [])
      setTasks(tRes?.tasks || tRes?.data?.tasks || [])
      if (pRes?.items) setPredictionRecords(pRes.items)
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

  const ck = useCohortAndKmFromPredictions({
    effectiveTaskId,
    effectiveModelType,
    cancer,
    availableTasks,
    tasks,
  })

  const barData = useMemo(() => {
    const x = result?.visualization?.probabilityBar?.x || []
    const y = result?.visualization?.probabilityBar?.y || []
    return x.map((name, i) => ({ name, p: y[i] ?? 0, fill: getBarColorByName(name) }))
  }, [result])

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
      const res = await predictApi.predict({
        caseId,
        taskId: effectiveTaskId,
        saveHistory: true,
        ensembleTiebreakAllowFallback: readEnsembleTiebreakAllowFallback(),
      })
      setResult(res)
      setNotice('预测完成')
      setPredictProgress(100)
      await ck.reloadCohortAndKm()
      await loadPredictionRecords()
    } catch (e) {
      setError(e?.response?.data?.message || e.message || '预测失败')
    } finally {
      clearInterval(timer)
      setLoading(false)
      setTimeout(() => setPredictProgress(0), 800)
    }
  }

  const allCasesWithFeatures = useMemo(
    () => (cases || []).filter((c) => c.feature20FileId && c.feature10FileId),
    [cases]
  )

  const handleOpenBatchDialog = () => {
    setBatchSelectedCases(new Set())
    setBatchRangeFrom('1')
    setBatchRangeTo(String(Math.min(10, allCasesWithFeatures.length)))
    setBatchCaseDialogOpen(true)
  }

  const doBatchPredictMulti = async () => {
    const selectedCases = allCasesWithFeatures.filter((c) => batchSelectedCases.has(c.caseId))
    if (!selectedCases.length) {
      setError('请至少选择一个病例')
      return
    }
    const eid = String(effectiveTaskId || '').trim()
    if (!eid) {
      setError('请先选择一个训练任务')
      return
    }
    setBatchCaseDialogOpen(false)
    setLoading(true)
    setPredictProgress(1)
    setError('')
    setResult(null)
    const t0 = Date.now()
    const timer = setInterval(() => {
      const dt = Date.now() - t0
      const target = dt < 60_000 ? 20 : dt < 300_000 ? 55 : dt < 600_000 ? 80 : 92
      setPredictProgress((p) => (p < target ? p + 1 : p))
    }, 1200)
    try {
      const fb = readEnsembleTiebreakAllowFallback()
      const initialItems = selectedCases.map((c) => ({
        caseId: c.caseId, taskId: eid, saveHistory: true, ensembleTiebreakAllowFallback: fb,
      }))
      const data = await predictApi.predictBatch(initialItems)
      const merged = new Map()
      for (const row of Array.isArray(data?.results) ? data.results : []) {
        const k = `${String(row?.input?.caseId || '')}::${String(row?.input?.taskId || '')}`
        if (k !== '::') merged.set(k, row)
      }
      const failed = []
      for (const it of initialItems) {
        const k = `${String(it.caseId)}::${String(it.taskId)}`
        const row = merged.get(k)
        const out = row?.output
        if (!row || row.error || !out || !Number.isFinite(Number(out.riskScore))) failed.push(it)
      }
      let didRetry = false
      if (failed.length) {
        didRetry = true
        const data2 = await predictApi.predictBatch(failed)
        for (const row of Array.isArray(data2?.results) ? data2.results : []) {
          const k2 = `${String(row?.input?.caseId || '')}::${String(row?.input?.taskId || '')}`
          if (k2 !== '::') merged.set(k2, row)
        }
      }
      let ok = 0
      let fail = 0
      for (const it of initialItems) {
        const k = `${String(it.caseId)}::${String(it.taskId)}`
        const row = merged.get(k)
        if (row?.error) { fail += 1; continue }
        const out = row?.output
        if (out && typeof out === 'object' && out.message && !Number.isFinite(out.riskScore)) fail += 1
        else if (out && Number.isFinite(out.riskScore)) ok += 1
        else fail += 1
      }
      const retryHint = didRetry ? '（已对首轮失败病例自动重试一次）' : ''
      setNotice(`批量预测完成：成功 ${ok}，失败 ${fail}（共 ${initialItems.length} 条，${selectedCases.length} 病例 × 1 任务）${retryHint}`)
      setPredictProgress(100)
      await ck.reloadCohortAndKm()
      await loadPredictionRecords()
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
      </Box>

      {(ck.cohortCIndexByTask && ck.cohortCIndexByTask.length > 0) || ck.cohortCIndexAll?.cIndexSuppressedZh ? (
        <Card sx={{ mb: 3, borderRadius: 2, border: '1px solid', borderColor: 'divider' }}>
          <CardHeader title="历史预测队列 · 生存 C-index / HR（按模型）" />
          <CardContent>
            {ck.cohortSummaryForSelectedTaskDisplay ? (
              <Box sx={{ mb: 2, p: 1.5, borderRadius: 1, bgcolor: (theme) => alpha(theme.palette.primary.main, 0.06), border: '1px solid', borderColor: 'divider' }}>
                <Typography variant="subtitle2" sx={{ fontWeight: 700 }}>
                  当前所选任务
                </Typography>
                <Typography variant="caption" color="text.secondary" sx={{ display: 'block', mt: 0.25 }}>
                  taskId: <code>{String(ck.cohortSummaryForSelectedTaskDisplay.taskId || effectiveTaskId || '')}</code> ·{' '}
                  {ck.resolveCohortTaskLabel(ck.cohortSummaryForSelectedTaskDisplay)}
                </Typography>
                <Typography variant="body2" sx={{ mt: 0.75 }}>
                  队列 C-index:{' '}
                  <strong>
                    {ck.cohortSummaryForSelectedTaskDisplay.cIndex != null
                      ? Number(ck.cohortSummaryForSelectedTaskDisplay.cIndex).toFixed(4)
                      : '—'}
                  </strong>
                  {ck.cohortSummaryForSelectedTaskDisplay.cIndex != null
                    ? formatCindex95CiParen(ck.cohortSummaryForSelectedTaskDisplay.cIndex95Ci)
                    : null}
                  {' · '}
                  可用病例 n={ck.cohortSummaryForSelectedTaskDisplay.nUsableCasesJoinedClinical ?? '—'}，可比患者对=
                  {ck.cohortSummaryForSelectedTaskDisplay.comparablePairs ?? '—'}
                </Typography>
                <Typography variant="body2" sx={{ mt: 0.5 }}>
                  Cox HR: <strong>{cohortHazardRatioCellText(ck.cohortSummaryForSelectedTaskDisplay)}</strong>
                </Typography>
                {ck.cohortSummaryForSelectedTaskDisplay.cIndexBootstrapNoteZh ? (
                  <Typography variant="caption" color="warning.main" sx={{ display: 'block', mt: 0.5 }}>
                    {ck.cohortSummaryForSelectedTaskDisplay.cIndexBootstrapNoteZh}
                  </Typography>
                ) : null}
              </Box>
            ) : null}

            <Typography variant="subtitle2" sx={{ fontWeight: 700, mb: 0.5 }}>
              各模型队列 C-index
            </Typography>
            {ck.cohortCIndexByTask.length === 0 ? (
              <Typography variant="body2" color="text.secondary">
                暂无带 <code>taskId</code> 的预测记录。请先在下方选择任务并完成至少一次 Predict；表格会在刷新页面或预测成功后自动更新。
              </Typography>
            ) : (
              <>
                {ck.cohortCindexRowsForDisplay.length === 0 ? (
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
                        <TableCell align="right">HR（Cox）</TableCell>
                        <TableCell align="right">可用病例 n</TableCell>
                        <TableCell align="right">可比患者对</TableCell>
                      </TableRow>
                    </TableHead>
                    <TableBody>
                      {(() => {
                        const focusMt = String(effectiveModelType || '').trim()
                        const dimOtherModels = Boolean(String(effectiveTaskId || '').trim() && focusMt)
                        return ck.cohortCindexRowsFixedSixDisplay.map((row) => {
                        const sel = Boolean(
                          row.taskId && effectiveTaskId && String(row.taskId) === String(effectiveTaskId)
                        )
                        const isBest =
                          row.taskId &&
                          ck.cohortCindexBestAmongDisplay &&
                          String(row.taskId) === String(ck.cohortCindexBestAmongDisplay.taskId) &&
                          String(row.modelType) === String(ck.cohortCindexBestAmongDisplay.modelType)
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
                                  <Chip size="small" variant="outlined" label="队列代表" />
                                ) : focusMt ? (
                                  <Chip size="small" color="info" variant="outlined" label="随底部任务" />
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
                                  {ck.resolveCohortTaskLabel(row)}
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
                            <TableCell align="right" sx={{ fontFamily: 'ui-monospace, monospace', fontSize: 11, maxWidth: 200, whiteSpace: 'normal', wordBreak: 'break-word' }}>
                              {cohortHazardRatioCellText(row)}
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
        <CardHeader
          title={
            <Box
              component="span"
              sx={{ cursor: 'pointer', userSelect: 'none', display: 'flex', alignItems: 'center', gap: 1 }}
              onClick={() => setHistoryExpanded((v) => !v)}
            >
              {historyExpanded ? '▼' : '▶'} 预测历史记录管理
            </Box>
          }
          action={
            <Box sx={{ display: 'flex', gap: 1, alignItems: 'center' }}>
              <Button
                size="small"
                variant="outlined"
                onClick={loadPredictionRecords}
                disabled={predictionRecordsLoading}
              >
                刷新
              </Button>
              <Button
                size="small"
                variant="outlined"
                color="error"
                onClick={handleDeleteSelectedRecords}
                disabled={selectedRecordIds.size === 0}
              >
                删除选中 ({selectedRecordIds.size})
              </Button>
              <Button
                size="small"
                variant="contained"
                color="error"
                onClick={handleDeleteAllRecords}
                disabled={predictionRecords.length === 0}
              >
                一键清空
              </Button>
            </Box>
          }
        />
        <Collapse in={historyExpanded}>
        <CardContent>
          {predictionRecordsLoading ? (
            <Box sx={{ display: 'flex', justifyContent: 'center', py: 3 }}>
              <CircularProgress size={28} />
            </Box>
          ) : predictionRecords.length === 0 ? (
            <Typography variant="body2" color="text.secondary">
              暂无预测历史记录。完成 Predict 后，记录会自动保存到此。
            </Typography>
          ) : (
            <>
              <Box sx={{ display: 'flex', alignItems: 'center', gap: 1, mb: 1 }}>
                <Button size="small" variant="text" onClick={handleSelectAllRecords}>
                  {selectedRecordIds.size === predictionRecords.length ? '取消全选' : '全选'}
                </Button>
                <Typography variant="caption" color="text.secondary">
                  共 {predictionRecords.length} 条记录，已选 {selectedRecordIds.size} 条
                </Typography>
              </Box>
              <TableContainer component={Paper} variant="outlined" sx={{ maxHeight: 400 }}>
                <Table size="small" stickyHeader>
                  <TableHead>
                    <TableRow>
                      <TableCell padding="checkbox" sx={{ width: 48 }} />
                      <TableCell>Case ID</TableCell>
                      <TableCell>Task ID</TableCell>
                      <TableCell align="right">Risk Score</TableCell>
                      <TableCell>风险分层</TableCell>
                      <TableCell>模型类型</TableCell>
                      <TableCell>预测时间</TableCell>
                    </TableRow>
                  </TableHead>
                  <TableBody>
                    {predictionRecords.map((rec) => {
                      const recId = rec.id || ''
                      const checked = selectedRecordIds.has(recId)
                      return (
                        <TableRow
                          key={recId}
                          hover
                          selected={checked}
                          sx={{ cursor: 'pointer' }}
                          onClick={() => handleSelectRecord(recId)}
                        >
                          <TableCell padding="checkbox" onClick={(e) => e.stopPropagation()}>
                            <Checkbox
                              checked={checked}
                              onChange={() => handleSelectRecord(recId)}
                              size="small"
                            />
                          </TableCell>
                          <TableCell sx={{ fontFamily: 'ui-monospace, monospace', fontSize: 12 }}>
                            {rec.caseId || '—'}
                          </TableCell>
                          <TableCell
                            sx={{ fontFamily: 'ui-monospace, monospace', fontSize: 12, maxWidth: 160, overflow: 'hidden', textOverflow: 'ellipsis', whiteSpace: 'nowrap' }}
                            title={rec.taskId}
                          >
                            {rec.taskId ? shortTaskId(rec.taskId) : '—'}
                          </TableCell>
                          <TableCell align="right" sx={{ fontFamily: 'ui-monospace, monospace', fontSize: 12 }}>
                            {Number.isFinite(rec.riskScore) ? Number(rec.riskScore).toFixed(4) : '—'}
                          </TableCell>
                          <TableCell>
                            {rec.riskStratification?.labelZh ? (
                              <RiskBadge tierZh={rec.riskStratification.labelZh} />
                            ) : '—'}
                          </TableCell>
                          <TableCell sx={{ fontSize: 12 }}>
                            {rec.modelType || '—'}
                          </TableCell>
                          <TableCell sx={{ fontSize: 12 }}>
                            {rec.createdAt
                              ? new Date(rec.createdAt).toLocaleString('zh-CN', {
                                  year: 'numeric',
                                  month: '2-digit',
                                  day: '2-digit',
                                  hour: '2-digit',
                                  minute: '2-digit',
                                })
                              : '—'}
                          </TableCell>
                        </TableRow>
                      )
                    })}
                  </TableBody>
                </Table>
              </TableContainer>
            </>
          )}
        </CardContent>
        </Collapse>
      </Card>

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
              onClick={handleOpenBatchDialog}
              disabled={loading || allCasesWithFeatures.length === 0}
            >
              批量预测（共 {allCasesWithFeatures.length} 病例）
            </Button>
          </Box>
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
            </CardContent>
          </Card>
        </>
      )}

      <Dialog open={batchCaseDialogOpen} onClose={() => setBatchCaseDialogOpen(false)} maxWidth="sm" fullWidth>
        <DialogTitle>选择批量预测病例</DialogTitle>
        <DialogContent>
          <Typography variant="body2" color="text.secondary" sx={{ mb: 1.5 }}>
            勾选一个或多个病例，将使用当前选中的任务（taskId: <code>{String(effectiveTaskId || '').slice(0, 12)}…</code>）对所选病例进行批量预测。
          </Typography>
          {allCasesWithFeatures.length === 0 ? (
            <Typography variant="body2" color="warning.main">
              暂无已绑定 20×/10× 特征的病例。
            </Typography>
          ) : (
            <><Stack direction="row" spacing={1} alignItems="center" sx={{ mb: 1.5 }}>
              <Typography variant="body2" color="text.secondary" sx={{ mr: 0.5, whiteSpace: 'nowrap' }}>
                快速勾选：
              </Typography>
              <Typography variant="body2" color="text.secondary" sx={{ whiteSpace: 'nowrap' }}>
                从第
              </Typography>
              <TextField
                type="number"
                size="small"
                slotProps={{ htmlInput: { min: 1, max: allCasesWithFeatures.length, style: { width: 56, textAlign: 'center' } } }}
                value={batchRangeFrom}
                onChange={(e) => setBatchRangeFrom(e.target.value)}
              />
              <Typography variant="body2" color="text.secondary" sx={{ whiteSpace: 'nowrap' }}>
                到第
              </Typography>
              <TextField
                type="number"
                size="small"
                slotProps={{ htmlInput: { min: 1, max: allCasesWithFeatures.length, style: { width: 56, textAlign: 'center' } } }}
                value={batchRangeTo}
                onChange={(e) => setBatchRangeTo(e.target.value)}
              />
              <Typography variant="body2" color="text.secondary" sx={{ whiteSpace: 'nowrap' }}>
                个病例
              </Typography>
              <Button
                size="small"
                variant="outlined"
                onClick={() => {
                  const from = Math.max(1, Number(batchRangeFrom) || 1)
                  const to = Math.min(allCasesWithFeatures.length, Math.max(from, Number(batchRangeTo) || from))
                  const selected = new Set(allCasesWithFeatures.slice(from - 1, to).map((x) => String(x.caseId || '').trim()).filter(Boolean))
                  setBatchSelectedCases(selected)
                }}
              >
                勾选
              </Button>
              <Button
                size="small"
                variant="text"
                onClick={() => {
                  const all = new Set(allCasesWithFeatures.map((x) => String(x.caseId || '').trim()).filter(Boolean))
                  setBatchSelectedCases(all)
                }}
              >
                全选
              </Button>
              <Button
                size="small"
                variant="text"
                color="error"
                onClick={() => setBatchSelectedCases(new Set())}
              >
                清空
              </Button>
            </Stack>
            <List dense disablePadding>
              {allCasesWithFeatures.map((c) => {
                const cid = String(c.caseId || '').trim()
                if (!cid) return null
                return (
                  <ListItem key={cid} disableGutters
                    secondaryAction={
                      <Checkbox
                        edge="end"
                        checked={batchSelectedCases.has(cid)}
                        onChange={() =>
                          setBatchSelectedCases((prev) => {
                            const next = new Set(prev)
                            if (next.has(cid)) next.delete(cid)
                            else next.add(cid)
                            return next
                          })
                        }
                      />
                    }
                    sx={{ borderRadius: 1, mb: 0.5, '&:hover': { bgcolor: 'action.hover' } }}
                  >
                    <ListItemText
                      primary={cid}
                      secondary={`cancer=${String(c.cancer || c.cancerType || '—')} · feature20FeatureId=${String(c.feature20FileId || '').slice(0, 8)}… feature10FeatureId=${String(c.feature10FileId || '').slice(0, 8)}…`}
                      primaryTypographyProps={{ variant: 'body2', fontWeight: 600, fontFamily: 'ui-monospace, monospace' }}
                      secondaryTypographyProps={{ variant: 'caption' }}
                    />
                  </ListItem>
                )
              })}
            </List></>
          )}
        </DialogContent>
        <DialogActions>
          <Button onClick={() => setBatchCaseDialogOpen(false)}>取消</Button>
          <Button
            variant="contained"
            onClick={doBatchPredictMulti}
            disabled={batchSelectedCases.size === 0 || !effectiveTaskId}
          >
            运行（{batchSelectedCases.size} 病例 × 1 任务）
          </Button>
        </DialogActions>
      </Dialog>
    </Box>
  )
}
