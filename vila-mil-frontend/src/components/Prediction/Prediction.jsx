import React, { useEffect, useMemo, useState } from 'react'
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
  Table,
  TableBody,
  TableCell,
  TableContainer,
  TableHead,
  TableRow,
  TextField,
  ToggleButton,
  ToggleButtonGroup,
  Typography,
} from '@mui/material'
import { alpha } from '@mui/material/styles'
import { Bar, BarChart, CartesianGrid, Cell, ResponsiveContainer, Tooltip, XAxis, YAxis } from 'recharts'
import { clinicalApi, dataApi, predictApi, trainingApi } from '../../services/api'
import useCancerOptions from '../../hooks/useCancerOptions'
import Toast from '../common/Toast.jsx'
import RasterPreview from '../Clinical/RasterPreview.jsx'

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

const cohortQueueCIndexText = (row) => {
  if (!row) return '—'
  if (row.cIndex != null) return Number(row.cIndex).toFixed(4)
  if (row.cIndexSuppressedZh) return '—'
  return '—'
}

export default function Prediction() {
  const [cases, setCases] = useState([])
  const [tasks, setTasks] = useState([])
  const [caseId, setCaseId] = useState('')
  const [taskId, setTaskId] = useState('')
  const [taskPickMode, setTaskPickMode] = useState('best') // best | manual
  const [pickedModelKey, setPickedModelKey] = useState('') // `${cancer}__${modelType}`
  const [bestTaskMeta, setBestTaskMeta] = useState(null)
  const inputMode = 'case'

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
      const data = await predictApi.listPredictions(250, {})
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
      if (inputMode === 'case' && caseFeatureMeta?.ready) {
        const firstCompatible = modelOptions.find((o) => modelCompatMap.get(o.key)?.compatibleAny)
        setPickedModelKey((firstCompatible || modelOptions[0]).key)
      } else {
        setPickedModelKey(modelOptions[0].key)
      }
    }
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [taskPickMode, modelOptions, inputMode, caseFeatureMeta])

  useEffect(() => {
    if (taskPickMode !== 'best') return
    if (!pickedModelKey) return
    const [c, m] = pickedModelKey.split('__')
    ;(async () => {
      setBestTaskMeta(null)
      const compatList = (availableTasks || []).filter((t) => {
        if (!(inputMode === 'case' && caseFeatureMeta?.ready)) return true
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
  }, [taskPickMode, pickedModelKey, availableTasks, inputMode, caseFeatureMeta])

  useEffect(() => {
    if (!taskId) return
    const ok = availableTasks.some((t) => t.taskId === taskId)
    if (!ok) setTaskId('')
  }, [taskId, availableTasks])

  useEffect(() => {
    if (inputMode !== 'case' || !caseId) {
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
  }, [inputMode, caseId])

  const getTaskCompatibility = (t, featureMeta) => {
    if (!featureMeta || !featureMeta.ready) return { compatible: true, reason: '' }
    const modelType = String(t?.modelType || t?.model_type || '')
    if (modelType === 'EnsembleFeature') {
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
  }

  const manualTaskOptions = useMemo(
    () =>
      (availableTasks || []).map((t) => {
        const check = getTaskCompatibility(t, inputMode === 'case' ? caseFeatureMeta : null)
        return { task: t, compatible: check.compatible, reason: check.reason }
      }),
    [availableTasks, inputMode, caseFeatureMeta]
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

  const compatibleTaskIdSet = useMemo(
    () => new Set(manualTaskOptions.filter((x) => x.compatible).map((x) => String(x.task.taskId || ''))),
    [manualTaskOptions]
  )

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

  const cohortCindexBestAmongDisplay = useMemo(() => {
    const rows = cohortCindexRowsByModel
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
  }, [cohortCindexRowsByModel])

  const effectiveTaskId = useMemo(() => {
    if (taskPickMode === 'best') {
      const bestId = String(bestTaskMeta?.bestTaskId || '')
      if (bestId) {
        if (!(inputMode === 'case' && caseFeatureMeta?.ready) || compatibleTaskIdSet.has(bestId)) return bestId
      }
      return taskId || ''
    }
    return taskId
  }, [taskPickMode, bestTaskMeta, taskId, inputMode, caseFeatureMeta, compatibleTaskIdSet])

  useEffect(() => {
    loadCohortCIndex()
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [tasks])

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
          选择训练任务后，按病例推理（使用 Clinical 中该病例已绑定的 20×/10× 特征）。
        </Typography>
      </Box>

      {(cohortCIndexAll || (cohortCIndexByTask && cohortCIndexByTask.length > 0)) && (
        <Card sx={{ mb: 3, borderRadius: 2, border: '1px solid', borderColor: 'divider' }}>
          <CardHeader title="历史预测队列 · 生存 C-index（按模型）" />
          <CardContent>
            <Typography variant="body2" color="text.secondary" sx={{ mb: 1.5 }}>
              <strong>怎么用：</strong>在 Clinical 填写随访 <code>time</code> / <code>status</code>，用各模型任务对多病例做{' '}
              <strong>Predict</strong> 写入历史。下表<strong>每个模型类型一行</strong>：只保留已能算出队列 C-index 的任务；若同模型有多个训练任务，取
              <strong>C-index 最高</strong>的那条并展示其 taskId。
            </Typography>
            <Typography variant="caption" color="text.secondary" sx={{ display: 'block', mb: 1.5 }}>
              注意：这与训练日志里的<strong>验证集 C-index</strong>不同；此处为「历史 riskScore + 随访」的队列一致性。
            </Typography>
            {cohortCIndexAll ? (
              <Box sx={{ mb: 2, p: 1.5, borderRadius: 1, bgcolor: (theme) => alpha(theme.palette.info.main, 0.06), border: '1px solid', borderColor: 'divider' }}>
                <Typography variant="subtitle2" sx={{ fontWeight: 700 }}>
                  全部任务合并（不区分 taskId）
                </Typography>
                <Typography variant="body2" sx={{ mt: 0.5 }}>
                  C-index:{' '}
                  <strong>
                    {cohortCIndexAll.cIndex != null
                      ? Number(cohortCIndexAll.cIndex).toFixed(4)
                      : cohortCIndexAll.cIndexSuppressedZh
                        ? '—'
                        : '—（可比样本不足）'}
                  </strong>
                  {' · '}
                  可用病例 n={cohortCIndexAll.nUsableCasesJoinedClinical ?? '—'}，可比患者对=
                  {cohortCIndexAll.comparablePairs ?? '—'}
                </Typography>
                {cohortCIndexAll.cIndexSuppressedZh ? (
                  <Typography variant="caption" color="text.secondary" sx={{ mt: 0.5, display: 'block' }}>
                    {cohortCIndexAll.cIndexSuppressedZh}
                  </Typography>
                ) : null}
              </Box>
            ) : null}

            <Typography variant="subtitle2" sx={{ fontWeight: 700, mb: 1 }}>
              各模型队列 C-index
            </Typography>
            {cohortCIndexByTask.length === 0 ? (
              <Typography variant="body2" color="text.secondary">
                暂无带 <code>taskId</code> 的预测记录。请先在下方选择任务并完成至少一次 Predict；表格会在刷新页面或预测成功后自动更新。
              </Typography>
            ) : cohortCindexRowsByModel.length === 0 ? (
              <Typography variant="body2" color="text.secondary">
                暂无各模型可计算的队列 C-index（多为随访不足或可比患者对为 0）。请补充 Clinical 的 <code>time</code>/<code>status</code> 并增加预测后再看。
              </Typography>
            ) : (
              <TableContainer component={Paper} variant="outlined" sx={{ maxHeight: 380 }}>
                <Table size="small" stickyHeader>
                  <TableHead>
                    <TableRow>
                      <TableCell>模型类型</TableCell>
                      <TableCell>代表任务 taskId</TableCell>
                      <TableCell align="right">队列 C-index</TableCell>
                      <TableCell align="right">可用病例 n</TableCell>
                      <TableCell align="right">可比患者对</TableCell>
                    </TableRow>
                  </TableHead>
                  <TableBody>
                    {cohortCindexRowsByModel.map((row) => {
                      const sel = Boolean(effectiveTaskId && String(row.taskId) === String(effectiveTaskId))
                      const isBest =
                        cohortCindexBestAmongDisplay &&
                        String(row.taskId) === String(cohortCindexBestAmongDisplay.taskId) &&
                        String(row.modelType) === String(cohortCindexBestAmongDisplay.modelType)
                      return (
                        <TableRow
                          key={`${row.modelType}-${row.taskId}`}
                          hover
                          selected={sel}
                          sx={
                            sel
                              ? (theme) => ({
                                  bgcolor: alpha(theme.palette.primary.main, theme.palette.mode === 'dark' ? 0.22 : 0.1),
                                })
                              : undefined
                          }
                        >
                          <TableCell sx={{ fontWeight: 600 }}>{row.modelType ?? '—'}</TableCell>
                          <TableCell title={row.taskId}>
                            <Box sx={{ display: 'flex', alignItems: 'center', gap: 0.75, flexWrap: 'wrap' }}>
                              <Typography component="span" sx={{ fontFamily: 'ui-monospace, monospace', fontSize: 12 }}>
                                {shortTaskId(row.taskId)}
                              </Typography>
                              {isBest ? <Chip size="small" color="success" label="全局最高" /> : null}
                              {sel ? <Chip size="small" color="primary" label="当前选中" /> : null}
                            </Box>
                            {row.taskLabel ? (
                              <Typography variant="caption" color="text.secondary" display="block">
                                {row.taskLabel}
                              </Typography>
                            ) : null}
                          </TableCell>
                          <TableCell align="right">
                            <strong>{Number(row.cIndex).toFixed(4)}</strong>
                          </TableCell>
                          <TableCell align="right">{row.nUsableCasesJoinedClinical ?? '—'}</TableCell>
                          <TableCell align="right">{row.comparablePairs ?? '—'}</TableCell>
                        </TableRow>
                      )
                    })}
                  </TableBody>
                </Table>
              </TableContainer>
            )}
          </CardContent>
        </Card>
      )}

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
                        disabled={inputMode === 'case' && caseFeatureMeta?.ready && !modelCompatMap.get(o.key)?.compatibleAny}
                        title={
                          inputMode === 'case' && caseFeatureMeta?.ready && !modelCompatMap.get(o.key)?.compatibleAny
                            ? modelCompatMap.get(o.key)?.reason || '该模型下暂无与当前 case 维度匹配的任务'
                            : ''
                        }
                      >
                        {o.modelType} — {o.cancer}
                        {inputMode === 'case' && caseFeatureMeta?.ready && !modelCompatMap.get(o.key)?.compatibleAny
                          ? '（维度不匹配）'
                          : ''}
                      </MenuItem>
                    ))}
                  </Select>
                </FormControl>
              )}

              {taskPickMode === 'manual' && (
                <FormControl sx={{ minWidth: 360 }}>
                  <InputLabel id="task-label">Task</InputLabel>
                  <Select labelId="task-label" label="Task" value={taskId} onChange={(e) => setTaskId(e.target.value)}>
                    {manualTaskOptions.map(({ task: t, compatible, reason }) => (
                      <MenuItem
                        key={t.taskId}
                        value={t.taskId}
                        disabled={inputMode === 'case' && caseFeatureMeta?.ready && !compatible}
                        title={!compatible ? reason : t.taskId}
                      >
                        {t.modelType} — {t.cancer} — epochs:{t.maxEpochs} — ckpt:{t.checkpointCount ?? 0}
                        {inputMode === 'case' && caseFeatureMeta?.ready && !compatible ? '（维度不匹配）' : ''}
                      </MenuItem>
                    ))}
                  </Select>
                </FormControl>
              )}
            </Box>

            <Button variant="outlined" onClick={load}>
              Refresh
            </Button>
            <Button variant="contained" onClick={doPredict} disabled={loading || !effectiveTaskId}>
              {loading ? <CircularProgress size={20} color="inherit" /> : 'Predict'}
            </Button>
          </Box>
          <Typography variant="caption" color="text.secondary" sx={{ mt: 1, display: 'block' }}>
            仅显示“已完成且存在 checkpoint”的任务。预测将使用当前病例在 Clinical 中已关联的双尺度特征。
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
