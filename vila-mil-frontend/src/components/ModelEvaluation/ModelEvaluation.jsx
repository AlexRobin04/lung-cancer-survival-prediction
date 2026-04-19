import React, { useEffect, useMemo, useState } from 'react'
import {
  Alert,
  Box,
  Button,
  Card,
  Checkbox,
  CardContent,
  CardHeader,
  CircularProgress,
  FormControl,
  FormControlLabel,
  InputLabel,
  MenuItem,
  Select,
  Stack,
  Typography,
  Grid,
  ToggleButton,
  ToggleButtonGroup,
  Collapse,
  Table,
  TableBody,
  TableCell,
  TableContainer,
  TableHead,
  TableRow,
  Paper,
} from '@mui/material'
import { alpha } from '@mui/material/styles'
import {
  CartesianGrid,
  Legend,
  Line,
  LineChart,
  ResponsiveContainer,
  Tooltip,
  XAxis,
  YAxis,
} from 'recharts'
import { evaluationApi, trainingApi } from '../../services/api'

const sectionCardSx = (accent, { mb = 3, mt } = {}) => (theme) => ({
  mb,
  ...(mt != null ? { mt } : {}),
  borderRadius: 2,
  overflow: 'hidden',
  border: '1px solid',
  borderColor: 'divider',
  borderTop: '4px solid',
  borderTopColor: accent,
  bgcolor: alpha(accent, theme.palette.mode === 'dark' ? 0.12 : 0.05),
  boxShadow:
    theme.palette.mode === 'dark' ? 'none' : `0 4px 16px ${alpha(accent, 0.1)}`,
  transition: 'box-shadow 0.2s ease',
  '&:hover': {
    boxShadow:
      theme.palette.mode === 'dark' ? 'none' : `0 8px 22px ${alpha(accent, 0.14)}`,
  },
})

const chartPanelSx = (theme) => ({
  height: 300,
  p: 1,
  borderRadius: 1.5,
  border: '1px solid',
  borderColor: 'divider',
  bgcolor: theme.palette.mode === 'dark' ? alpha(theme.palette.common.white, 0.02) : '#fbfcff',
})

const fmt = (v, digits = 4) => {
  if (v === null || typeof v === 'undefined' || v === '') return '—'
  const n = Number(v)
  return Number.isFinite(n) ? n.toFixed(digits) : '—'
}

const legendName = {
  trainLoss: '训练集 Loss',
  valLoss: '验证集 Loss',
}
const COMPARE_MODELS = ['RRTMIL', 'AMIL', 'WiKG', 'DSMIL', 'S4MIL', 'EnsembleFeature']
const COMPARE_COLORS = {
  RRTMIL: '#1565c0',
  AMIL: '#2e7d32',
  WiKG: '#ef6c00',
  DSMIL: '#6a1b9a',
  S4MIL: '#00838f',
  EnsembleFeature: '#c62828',
}
const COMPARE_DASH = {
  RRTMIL: '0',
  AMIL: '6 3',
  WiKG: '2 2',
  DSMIL: '8 4',
  S4MIL: '10 4 2 4',
  EnsembleFeature: '0',
}

const toCsv = (rows, headers) => {
  const esc = (v) => {
    const s = v === null || typeof v === 'undefined' ? '' : String(v)
    if (/[\",\n]/.test(s)) return `"${s.replace(/\"/g, '""')}"`
    return s
  }
  const out = []
  out.push(headers.map(esc).join(','))
  for (const r of rows) out.push(headers.map((h) => esc(r[h])).join(','))
  return out.join('\n')
}

const downloadText = (filename, text) => {
  const blob = new Blob([text], { type: 'text/csv;charset=utf-8' })
  const url = URL.createObjectURL(blob)
  const a = document.createElement('a')
  a.href = url
  a.download = filename
  document.body.appendChild(a)
  a.click()
  a.remove()
  URL.revokeObjectURL(url)
}

/** 阶梯 KM：在时刻 t 的生存概率（由 lifelines 输出的 times/survival 还原） */
function kmSurvivalAt(times, survival, t) {
  const ts = times || []
  const ss = survival || []
  if (ts.length === 0) return 1
  let s = 1
  for (let i = 0; i < ts.length; i += 1) {
    if (ts[i] > t) break
    s = ss[i]
  }
  return s
}

/** 合并两条 KM 曲线为 Recharts 共用横轴数据 */
function mergeLuscKmRows(curveOurs, curveOthers) {
  const t1 = curveOurs?.times || []
  const s1 = curveOurs?.survival || []
  const t2 = curveOthers?.times || []
  const s2 = curveOthers?.survival || []
  const all = [...new Set([0, ...t1, ...t2])].sort((a, b) => a - b)
  return all.map((time) => ({
    time,
    ours: kmSurvivalAt(t1, s1, time),
    others: kmSurvivalAt(t2, s2, time),
  }))
}

export default function ModelEvaluation() {
  const [error, setError] = useState('')
  const [runs, setRuns] = useState([])
  const [taskId, setTaskId] = useState('')
  const [selectMode, setSelectMode] = useState('best') // best | manual
  const [yScaleMode, setYScaleMode] = useState('auto') // auto | zero
  const [bestModelKey, setBestModelKey] = useState('') // `${cancer}__${modelType}`
  const [bestInfo, setBestInfo] = useState(null)
  const [bestRefreshedAt, setBestRefreshedAt] = useState('')
  const [curves, setCurves] = useState(null)
  /** LUSC 示例 KM：后端 lusc (1).csv */
  const [luscKm, setLuscKm] = useState(null)
  const [luscKmLoading, setLuscKmLoading] = useState(true)
  const [luscKmError, setLuscKmError] = useState('')
  const [luscBaseline, setLuscBaseline] = useState('others') // others | ours
  const [showEpochTable, setShowEpochTable] = useState(false)
  const [compareCancer, setCompareCancer] = useState('LUSC')
  const [compareLoading, setCompareLoading] = useState(false)
  const [compareSeries, setCompareSeries] = useState([])
  const [compareMeta, setCompareMeta] = useState([])
  const [compareMetric, setCompareMetric] = useState('valLoss') // valLoss | trainLoss
  const [visibleCompareModels, setVisibleCompareModels] = useState(COMPARE_MODELS)
  const [compareMonotonicDisplay, setCompareMonotonicDisplay] = useState(false)

  const pickBestCompletedRun = (list, cancer, modelType) => {
    const cands = (list || []).filter(
      (r) => String(r?.status || '').toLowerCase() === 'completed' && String(r?.cancer || '') === cancer && String(r?.modelType || '') === modelType
    )
    if (cands.length === 0) return null
    const byLoss = [...cands].sort((a, b) => {
      const la = Number(a?.loss)
      const lb = Number(b?.loss)
      const va = Number.isFinite(la) ? la : Number.POSITIVE_INFINITY
      const vb = Number.isFinite(lb) ? lb : Number.POSITIVE_INFINITY
      if (va !== vb) return va - vb
      return String(b?.taskId || '').localeCompare(String(a?.taskId || ''))
    })
    return byLoss[0] || cands[0]
  }

  const loadRuns = async () => {
    setError('')
    try {
      const r = await evaluationApi.runs()
      setRuns(r?.runs || [])
    } catch (e) {
      setError(e?.response?.data?.message || e.message || '加载 runs 失败')
      setRuns([])
    }
  }

  const loadCurves = async () => {
    if (!taskId) return
    setError('')
    try {
      const c = await evaluationApi.curves(taskId)
      setCurves(c)
    } catch (e) {
      setError(e?.response?.data?.message || e.message || '加载曲线失败')
      setCurves(null)
    }
  }

  useEffect(() => {
    loadRuns()
  }, [])

  useEffect(() => {
    let cancelled = false
    ;(async () => {
      setLuscKmError('')
      setLuscKmLoading(true)
      try {
        const d = await evaluationApi.kmLuscDemo()
        if (!cancelled) setLuscKm(d)
      } catch (e) {
        if (!cancelled) {
          setLuscKm(null)
          setLuscKmError(e?.response?.data?.message || e.message || '加载 LUSC 示例 KM 失败')
        }
      } finally {
        if (!cancelled) setLuscKmLoading(false)
      }
    })()
    return () => {
      cancelled = true
    }
  }, [])

  useEffect(() => {
    loadCurves()
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [taskId])

  useEffect(() => {
    if (!taskId) return
    let stopped = false
    const tick = async () => {
      try {
        const [c, r] = await Promise.all([evaluationApi.curves(taskId), evaluationApi.runs()])
        if (stopped) return
        setCurves(c)
        setRuns(r?.runs || [])
      } catch {
        // ignore polling errors
      }
    }
    const t = setInterval(tick, 5000)
    return () => {
      stopped = true
      clearInterval(t)
    }
  }, [taskId])

  const series = useMemo(() => curves?.series || [], [curves])
  const lossDomain = useMemo(() => {
    if (yScaleMode !== 'auto') return undefined
    const vals = []
    for (const p of series || []) {
      const a = Number(p?.trainLoss)
      const b = Number(p?.valLoss)
      if (Number.isFinite(a)) vals.push(a)
      if (Number.isFinite(b)) vals.push(b)
    }
    if (vals.length === 0) return undefined
    const min = Math.min(...vals)
    const max = Math.max(...vals)
    const span = max - min
    const pad = Math.max(span * 0.12, Math.abs(max) * 0.02, 1e-6)
    const lo = min - pad
    const hi = max + pad
    return [lo, hi]
  }, [series, yScaleMode])
  const completedRuns = useMemo(
    () => (runs || []).filter((r) => String(r?.status || '').toLowerCase() === 'completed'),
    [runs]
  )
  const compareCancerOptions = useMemo(() => {
    const set = new Set()
    for (const r of completedRuns || []) {
      const c = String(r?.cancer || '').trim()
      if (c) set.add(c)
    }
    const arr = Array.from(set).sort()
    return arr.length ? arr : ['LUSC']
  }, [completedRuns])
  const bestModelOptions = useMemo(() => {
    const map = new Map()
    for (const r of completedRuns || []) {
      const c = String(r?.cancer || '').trim()
      const m = String(r?.modelType || '').trim()
      if (!c || !m) continue
      const k = `${c}__${m}`
      if (!map.has(k)) map.set(k, { key: k, cancer: c, modelType: m })
    }
    return Array.from(map.values()).sort((a, b) =>
      `${a.cancer}-${a.modelType}`.localeCompare(`${b.cancer}-${b.modelType}`)
    )
  }, [completedRuns])
  const luscKmChartData = useMemo(() => {
    const curves = luscKm?.curves || []
    const ours = curves.find((c) => String(c?.label || '').toLowerCase() === 'ours')
    const others = curves.find((c) => String(c?.label || '').toLowerCase() === 'others')
    if (!ours || !others) return []
    return mergeLuscKmRows(ours, others)
  }, [luscKm])
  const compareDisplaySeries = useMemo(() => {
    if (!compareMonotonicDisplay) return compareSeries
    const out = (compareSeries || []).map((r) => ({ ...r }))
    for (const model of COMPARE_MODELS) {
      for (const metric of ['valLoss', 'trainLoss']) {
        const k = `${model}_${metric}`
        let best = Number.POSITIVE_INFINITY
        for (let i = 0; i < out.length; i += 1) {
          const v = Number(out[i]?.[k])
          if (!Number.isFinite(v)) continue
          if (v < best) best = v
          out[i][k] = best
        }
      }
    }
    return out
  }, [compareSeries, compareMonotonicDisplay])

  useEffect(() => {
    if (selectMode !== 'best') return
    if (!bestModelKey && bestModelOptions.length > 0) setBestModelKey(bestModelOptions[0].key)
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [selectMode, bestModelOptions])

  useEffect(() => {
    if (selectMode !== 'best') return
    if (!bestModelKey) return
    const [cancer, modelType] = bestModelKey.split('__')
    ;(async () => {
      setError('')
      setBestInfo(null)
      const localBest = pickBestCompletedRun(completedRuns, cancer, modelType)
      if (localBest?.taskId) setTaskId(localBest.taskId)
      try {
        const res = await trainingApi.best({ cancer, modelType, mode: 'transformer' })
        setBestInfo(res)
        setBestRefreshedAt(new Date().toLocaleString())
        const bestId = String(res?.bestTaskId || '')
        const inCompleted = (completedRuns || []).some((r) => String(r?.taskId || '') === bestId)
        if (bestId && inCompleted) setTaskId(bestId)
      } catch (e) {
        setBestInfo(null)
        setBestRefreshedAt('')
        if (!localBest?.taskId) setError(e?.response?.data?.message || e.message || '未找到可用的已完成训练记录')
      }
    })()
  }, [selectMode, bestModelKey, completedRuns])

  useEffect(() => {
    if (!compareCancerOptions.includes(compareCancer)) {
      setCompareCancer(compareCancerOptions[0] || 'LUSC')
    }
  }, [compareCancerOptions, compareCancer])

  useEffect(() => {
    let cancelled = false
    ;(async () => {
      setCompareLoading(true)
      try {
        const loaded = []
        for (const modelType of COMPARE_MODELS) {
          let targetTaskId = ''
          try {
            const best = await trainingApi.best({ cancer: compareCancer, modelType, mode: 'transformer' })
            targetTaskId = String(best?.bestTaskId || '')
          } catch {
            // ignore and fallback to local pick
          }
          if (!targetTaskId) {
            const localBest = pickBestCompletedRun(completedRuns, compareCancer, modelType)
            targetTaskId = String(localBest?.taskId || '')
          }
          if (!targetTaskId) continue
          try {
            const curvesData = await evaluationApi.curves(targetTaskId)
            const s = Array.isArray(curvesData?.series) ? curvesData.series : []
            if (s.length === 0) continue
            loaded.push({
              modelType,
              taskId: targetTaskId,
              points: s,
              bestValLoss: curvesData?.summary?.bestValLoss,
              summary: curvesData?.summary || {},
            })
          } catch {
            // skip bad task
          }
        }
        if (cancelled) return
        const epochs = new Set()
        loaded.forEach((m) => m.points.forEach((p) => epochs.add(Number(p?.epoch))))
        const merged = Array.from(epochs)
          .filter((e) => Number.isFinite(e))
          .sort((a, b) => a - b)
          .map((epoch) => {
            const row = { epoch }
            for (const m of loaded) {
              const hit = m.points.find((p) => Number(p?.epoch) === epoch)
              row[`${m.modelType}_valLoss`] = Number.isFinite(Number(hit?.valLoss)) ? Number(hit?.valLoss) : null
              row[`${m.modelType}_trainLoss`] = Number.isFinite(Number(hit?.trainLoss)) ? Number(hit?.trainLoss) : null
            }
            return row
          })
        setCompareMeta(
          loaded.map((m) => ({
            modelType: m.modelType,
            taskId: m.taskId,
            bestValLoss: m.bestValLoss,
            summary: m.summary || {},
          }))
        )
        setCompareSeries(merged)
      } finally {
        if (!cancelled) setCompareLoading(false)
      }
    })()
    return () => {
      cancelled = true
    }
  }, [compareCancer, completedRuns])

  const exportCsv = () => {
    if (!taskId || series.length === 0) return
    const rows = series.map((p) => ({
      epoch: p.epoch,
      trainLoss: p.trainLoss ?? '',
      valLoss: p.valLoss ?? '',
    }))
    const headers = ['epoch', 'trainLoss', 'valLoss']
    const csv = toCsv(rows, headers)
    downloadText(`model-evaluation-${taskId}.csv`, csv)
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
              ? 'linear-gradient(115deg, rgba(123,31,162,0.22) 0%, rgba(2,136,209,0.12) 100%)'
              : 'linear-gradient(115deg, rgba(123,31,162,0.10) 0%, rgba(2,136,209,0.06) 100%)',
        })}
      >
        <Typography variant="h4" gutterBottom sx={{ fontWeight: 700, mb: 1 }}>
          模型评估（训练曲线）
        </Typography>
        <Typography variant="body2" color="text.secondary">
          这里展示的是后端从训练日志解析得到的训练曲线（Loss），并附带 LUSC 示例 KM 对比。
        </Typography>
      </Box>

      {error && (
        <Alert severity="error" sx={{ mb: 2 }} onClose={() => setError('')}>
          {error}
        </Alert>
      )}

      <Card sx={sectionCardSx('#7b1fa2')}>
        <CardHeader
          title="选择训练任务（Run）"
          titleTypographyProps={{ variant: 'subtitle1', fontWeight: 700 }}
          action={
            <Button variant="outlined" onClick={loadRuns}>
              刷新列表
            </Button>
          }
        />
        <CardContent>
          <Box sx={{ display: 'flex', flexWrap: 'wrap', gap: 2, alignItems: 'center' }}>
            <ToggleButtonGroup value={selectMode} exclusive size="small" onChange={(_, v) => v && setSelectMode(v)}>
              <ToggleButton value="best">最佳</ToggleButton>
              <ToggleButton value="manual">手动</ToggleButton>
            </ToggleButtonGroup>

            {selectMode === 'best' && (
              <FormControl sx={{ minWidth: 320 }}>
                <InputLabel id="best-model-label">模型</InputLabel>
                <Select
                  labelId="best-model-label"
                  label="模型"
                  value={bestModelKey}
                  onChange={(e) => setBestModelKey(e.target.value)}
                >
                  {bestModelOptions.map((o) => (
                    <MenuItem key={o.key} value={o.key}>
                      {o.modelType} — {o.cancer}
                    </MenuItem>
                  ))}
                </Select>
              </FormControl>
            )}

            {selectMode === 'manual' && (
              <FormControl sx={{ minWidth: 420, flex: 1 }}>
                <InputLabel id="run-label">任务</InputLabel>
                <Select labelId="run-label" label="任务" value={taskId} onChange={(e) => setTaskId(e.target.value)}>
                  {runs.map((r) => (
                    <MenuItem key={r.taskId} value={r.taskId}>
                      {r.modelType} — {r.cancer} — {r.status}
                    </MenuItem>
                  ))}
                </Select>
              </FormControl>
            )}
          </Box>

          {selectMode === 'best' && bestInfo?.bestTaskId && (
            <Typography variant="caption" color="text.secondary" sx={{ mt: 1, display: 'block' }}>
              bestTaskId: <code>{bestInfo.bestTaskId}</code>
              {bestInfo?.metric?.bestValLoss != null ? (
                <>
                  {' '}
                 ；bestValLoss: <code>{String(bestInfo.metric.bestValLoss)}</code>
                </>
              ) : null}
              {bestRefreshedAt ? (
                <>
                  {' '}
                  ；最后刷新: <code>{bestRefreshedAt}</code>
                </>
              ) : null}
            </Typography>
          )}
        </CardContent>
      </Card>

      <Card sx={sectionCardSx('#0288d1')}>
        <CardHeader
          title="训练曲线"
          titleTypographyProps={{ variant: 'subtitle1', fontWeight: 700 }}
          subheader="同一任务可能包含 k-fold；曲线横轴 epoch 为“全局 epoch”（fold × maxEpochs + epoch）"
          action={
            <Box sx={{ display: 'flex', gap: 1, flexWrap: 'wrap' }}>
              <ToggleButtonGroup
                value={yScaleMode}
                exclusive
                size="small"
                onChange={(_, v) => v && setYScaleMode(v)}
                aria-label="y-scale"
              >
                <ToggleButton value="auto">Y 轴自适应</ToggleButton>
                <ToggleButton value="zero">Y 轴从 0</ToggleButton>
              </ToggleButtonGroup>
              <Button variant="outlined" onClick={() => setShowEpochTable((v) => !v)} disabled={series.length === 0}>
                {showEpochTable ? '收起明细' : '展开明细'}
              </Button>
              <Button variant="contained" onClick={exportCsv} disabled={series.length === 0 || !taskId}>
                导出 CSV
              </Button>
            </Box>
          }
        />
        <CardContent>
          {series.length === 0 ? (
            <Alert severity="info">暂无曲线数据（请先选择 completed 的 run）</Alert>
          ) : (
            <>
              <Grid container spacing={2}>
                <Grid item xs={12} md={6}>
                  <Typography variant="subtitle2" sx={{ mb: 1, fontWeight: 700 }}>
                    训练集 Loss
                  </Typography>
                  <Box sx={(theme) => chartPanelSx(theme)}>
                    <ResponsiveContainer width="100%" height="100%">
                      <LineChart data={series}>
                        <CartesianGrid strokeDasharray="3 3" stroke={alpha('#90a4ae', 0.35)} />
                        <XAxis dataKey="epoch" />
                        <YAxis
                          domain={yScaleMode === 'auto' ? lossDomain : [0, 'auto']}
                          tickFormatter={(v) => fmt(v, 3)}
                          width={54}
                        />
                        <Tooltip
                          formatter={(value, name) => [fmt(value), legendName[name] || name]}
                          labelFormatter={(label) => `epoch ${label}`}
                          contentStyle={{
                            borderRadius: 10,
                            border: '1px solid #d7dee8',
                            boxShadow: '0 8px 18px rgba(15,23,42,0.12)',
                          }}
                        />
                        <Line
                          type="monotone"
                          dataKey="trainLoss"
                          name={legendName.trainLoss}
                          stroke="#1565c0"
                          strokeWidth={2}
                          dot={false}
                        />
                      </LineChart>
                    </ResponsiveContainer>
                  </Box>
                </Grid>
                <Grid item xs={12} md={6}>
                  <Typography variant="subtitle2" sx={{ mb: 1, fontWeight: 700 }}>
                    验证集 Loss
                  </Typography>
                  <Box sx={(theme) => chartPanelSx(theme)}>
                    <ResponsiveContainer width="100%" height="100%">
                      <LineChart data={series}>
                        <CartesianGrid strokeDasharray="3 3" stroke={alpha('#90a4ae', 0.35)} />
                        <XAxis dataKey="epoch" />
                        <YAxis
                          domain={yScaleMode === 'auto' ? lossDomain : [0, 'auto']}
                          tickFormatter={(v) => fmt(v, 3)}
                          width={54}
                        />
                        <Tooltip
                          formatter={(value, name) => [fmt(value), legendName[name] || name]}
                          labelFormatter={(label) => `epoch ${label}`}
                          contentStyle={{
                            borderRadius: 10,
                            border: '1px solid #d7dee8',
                            boxShadow: '0 8px 18px rgba(15,23,42,0.12)',
                          }}
                        />
                        <Line
                          type="monotone"
                          dataKey="valLoss"
                          name={legendName.valLoss}
                          stroke="#7b1fa2"
                          strokeWidth={2}
                          dot={false}
                        />
                      </LineChart>
                    </ResponsiveContainer>
                  </Box>
                </Grid>
              </Grid>

              <Collapse in={showEpochTable} timeout="auto" unmountOnExit>
                <Box sx={{ mt: 2 }}>
                  <Typography variant="subtitle2" sx={{ mb: 1 }}>
                    各 epoch 指标明细（可导出 CSV）
                  </Typography>
                  <TableContainer
                    component={Paper}
                    variant="outlined"
                    sx={(theme) => ({
                      borderRadius: 1.5,
                      bgcolor: theme.palette.mode === 'dark' ? 'action.hover' : 'background.paper',
                    })}
                  >
                    <Table size="small">
                      <TableHead>
                        <TableRow>
                          <TableCell>epoch</TableCell>
                          <TableCell align="right">trainLoss</TableCell>
                          <TableCell align="right">valLoss</TableCell>
                        </TableRow>
                      </TableHead>
                      <TableBody>
                        {series.map((p) => (
                          <TableRow key={p.epoch}>
                            <TableCell>{p.epoch}</TableCell>
                            <TableCell align="right">{fmt(p.trainLoss)}</TableCell>
                            <TableCell align="right">{fmt(p.valLoss)}</TableCell>
                          </TableRow>
                        ))}
                      </TableBody>
                    </Table>
                  </TableContainer>
                  <Typography variant="caption" color="text.secondary" sx={{ display: 'block', mt: 1 }}>
                    说明：epoch 为全局展开（fold × maxEpochs + epoch）。
                  </Typography>
                </Box>
              </Collapse>
            </>
          )}
        </CardContent>
      </Card>

      <Card sx={sectionCardSx('#5d4037')}>
        <CardHeader
          title="最优任务对比（5基模型 + EnsembleFeature）"
          titleTypographyProps={{ variant: 'subtitle1', fontWeight: 700 }}
          subheader="同一癌种下，自动挑选每个模型最佳任务并在一张图比较验证集 Loss 曲线"
          action={
            <Box sx={{ display: 'flex', gap: 1, alignItems: 'center' }}>
              <FormControl size="small" sx={{ minWidth: 140 }}>
                <InputLabel id="compare-cancer-label">Cancer</InputLabel>
                <Select
                  labelId="compare-cancer-label"
                  label="Cancer"
                  value={compareCancer}
                  onChange={(e) => setCompareCancer(e.target.value)}
                >
                  {compareCancerOptions.map((c) => (
                    <MenuItem key={c} value={c}>
                      {c}
                    </MenuItem>
                  ))}
                </Select>
              </FormControl>
              <Button variant="outlined" onClick={loadRuns}>
                刷新
              </Button>
            </Box>
          }
        />
        <CardContent>
          {compareLoading ? (
            <Box sx={{ display: 'flex', alignItems: 'center', gap: 1, py: 1 }}>
              <CircularProgress size={20} />
              <Typography variant="body2" color="text.secondary">
                正在加载最优曲线对比…
              </Typography>
            </Box>
          ) : compareSeries.length === 0 ? (
            <Alert severity="info">当前癌种暂无可对比曲线（需要各模型有已完成任务）</Alert>
          ) : (
            <>
              <Stack direction={{ xs: 'column', md: 'row' }} spacing={1.5} sx={{ mb: 1.5 }}>
                <ToggleButtonGroup value={compareMetric} exclusive size="small" onChange={(_, v) => v && setCompareMetric(v)}>
                  <ToggleButton value="valLoss">验证集 Loss</ToggleButton>
                  <ToggleButton value="trainLoss">训练集 Loss</ToggleButton>
                </ToggleButtonGroup>
                <FormControlLabel
                  control={
                    <Checkbox
                      size="small"
                      checked={compareMonotonicDisplay}
                      onChange={(e) => setCompareMonotonicDisplay(e.target.checked)}
                    />
                  }
                  label={<Typography variant="caption">单调平滑展示（仅显示层，不改真实数据）</Typography>}
                />
                <Box sx={{ display: 'flex', flexWrap: 'wrap', gap: 0.5 }}>
                  {COMPARE_MODELS.map((m) => (
                    <FormControlLabel
                      key={m}
                      sx={{ mr: 0.5 }}
                      control={
                        <Checkbox
                          size="small"
                          checked={visibleCompareModels.includes(m)}
                          onChange={(e) => {
                            setVisibleCompareModels((prev) => {
                              const s = new Set(prev)
                              if (e.target.checked) s.add(m)
                              else s.delete(m)
                              return COMPARE_MODELS.filter((x) => s.has(x))
                            })
                          }}
                        />
                      }
                      label={<Typography variant="caption">{m}</Typography>}
                    />
                  ))}
                </Box>
              </Stack>
              <Box sx={(theme) => ({ ...chartPanelSx(theme), height: 360 })}>
                <ResponsiveContainer width="100%" height="100%">
                  <LineChart data={compareDisplaySeries}>
                    <CartesianGrid strokeDasharray="3 3" stroke={alpha('#90a4ae', 0.35)} />
                    <XAxis dataKey="epoch" />
                    <YAxis tickFormatter={(v) => fmt(v, 3)} width={54} />
                    <Tooltip
                      formatter={(value, name) => [fmt(value), `${name} ${compareMetric === 'valLoss' ? '验证集 Loss' : '训练集 Loss'}`]}
                      labelFormatter={(label) => `epoch ${label}`}
                    />
                    <Legend />
                    {COMPARE_MODELS.filter((m) => visibleCompareModels.includes(m)).map((m) => (
                      <Line
                        key={m}
                        type="monotone"
                        dataKey={`${m}_${compareMetric}`}
                        name={m}
                        stroke={COMPARE_COLORS[m] || '#333'}
                        strokeWidth={m === 'EnsembleFeature' ? 3 : 2}
                        strokeDasharray={COMPARE_DASH[m] || '0'}
                        dot={false}
                        activeDot={{ r: 3 }}
                      />
                    ))}
                  </LineChart>
                </ResponsiveContainer>
              </Box>
              <Typography variant="caption" color="text.secondary" sx={{ mt: 1, display: 'block' }}>
                对比任务：
                {compareMeta.map((m) => ` ${m.modelType}(${m.taskId.slice(0, 8)}...)`).join('；')}
              </Typography>
              {compareMonotonicDisplay && (
                <Typography variant="caption" color="warning.main" sx={{ mt: 0.5, display: 'block' }}>
                  已启用单调平滑展示：曲线按 epoch 做累计最小值处理，仅用于可视化对比，不会修改后端原始指标。
                </Typography>
              )}
            </>
          )}
        </CardContent>
      </Card>

      <Card sx={sectionCardSx('#6d4c41')}>
        <CardHeader
          title="最优 5 基模型 + EnsembleFeature 指标总览"
          titleTypographyProps={{ variant: 'subtitle1', fontWeight: 700 }}
          subheader="展示当前癌种下 5 个基模型 + EnsembleFeature 的最佳任务关键指标"
        />
        <CardContent>
          {compareLoading ? (
            <Box sx={{ display: 'flex', alignItems: 'center', gap: 1, py: 1 }}>
              <CircularProgress size={20} />
              <Typography variant="body2" color="text.secondary">
                正在加载指标总览…
              </Typography>
            </Box>
          ) : compareMeta.length === 0 ? (
            <Alert severity="info">暂无指标数据（请先完成对应模型训练）</Alert>
          ) : (
            <TableContainer
              component={Paper}
              variant="outlined"
              sx={(theme) => ({
                borderRadius: 1.5,
                bgcolor: theme.palette.mode === 'dark' ? 'action.hover' : 'background.paper',
              })}
            >
              <Table size="small">
                <TableHead>
                  <TableRow>
                    <TableCell>模型</TableCell>
                    <TableCell>bestTaskId</TableCell>
                    <TableCell align="right">bestValLoss</TableCell>
                    <TableCell align="right">finalValCIndex</TableCell>
                    <TableCell align="right">finalTestCIndex</TableCell>
                    <TableCell align="right">epochCount</TableCell>
                  </TableRow>
                </TableHead>
                <TableBody>
                  {COMPARE_MODELS.map((modelType) => {
                    const row = compareMeta.find((x) => x.modelType === modelType)
                    if (!row) {
                      return (
                        <TableRow key={modelType}>
                          <TableCell>{modelType}</TableCell>
                          <TableCell colSpan={5} sx={{ color: 'text.secondary' }}>
                            暂无最佳任务
                          </TableCell>
                        </TableRow>
                      )
                    }
                    const s = row.summary || {}
                    return (
                      <TableRow key={modelType}>
                        <TableCell sx={{ fontWeight: modelType === 'EnsembleFeature' ? 700 : 500 }}>{modelType}</TableCell>
                        <TableCell>
                          <code>{String(row.taskId || '').slice(0, 12)}...</code>
                        </TableCell>
                        <TableCell align="right">{fmt(s.bestValLoss ?? row.bestValLoss)}</TableCell>
                        <TableCell align="right">{fmt(s.finalValCIndex)}</TableCell>
                        <TableCell align="right">{fmt(s.finalTestCIndex)}</TableCell>
                        <TableCell align="right">{Number.isFinite(Number(s.epochCount)) ? Number(s.epochCount) : '—'}</TableCell>
                      </TableRow>
                    )
                  })}
                </TableBody>
              </Table>
            </TableContainer>
          )}
          <Typography variant="caption" color="text.secondary" sx={{ mt: 1, display: 'block' }}>
            注：EnsembleFeature 与单模型任务相同，由该任务训练日志解析曲线与指标；若日志字段缺失，部分列可能显示为「—」。
          </Typography>
        </CardContent>
      </Card>

      <Card sx={sectionCardSx('#2e7d32', { mb: 0, mt: 3 })}>
        <CardHeader
          title="生存评估：LUSC 示例（Ours vs Others）"
          titleTypographyProps={{ variant: 'subtitle1', fontWeight: 700 }}
          subheader="数据来自后端 lusc (1).csv；baseline 仅影响图例线型（虚线=参照组）"
        />
        <CardContent>
          {luscKmLoading && (
            <Box sx={{ display: 'flex', alignItems: 'center', gap: 1, py: 2 }}>
              <CircularProgress size={22} />
              <Typography variant="body2" color="text.secondary">
                正在加载 LUSC 示例 KM…
              </Typography>
            </Box>
          )}
          {luscKmError && (
            <Alert severity="warning" sx={{ mb: 2 }}>
              {luscKmError}
            </Alert>
          )}
          {!luscKmLoading && luscKm && !luscKmError && (
            <>
              <Box sx={{ display: 'flex', flexWrap: 'wrap', gap: 2, alignItems: 'center', mb: 2 }}>
                <FormControl sx={{ minWidth: 220 }} size="small">
                  <InputLabel id="lusc-baseline-label">Baseline（参照曲线）</InputLabel>
                  <Select
                    labelId="lusc-baseline-label"
                    label="Baseline（参照曲线）"
                    value={luscBaseline}
                    onChange={(e) => setLuscBaseline(e.target.value)}
                  >
                    <MenuItem value="others">Others 为 baseline</MenuItem>
                    <MenuItem value="ours">Ours 为 baseline</MenuItem>
                  </Select>
                </FormControl>
                <Typography variant="body2" color="text.secondary">
                  log-rank p（互斥子集）:{' '}
                  <strong>
                    {luscKm.logRankPExclusive != null ? Number(luscKm.logRankPExclusive).toFixed(6) : '—'}
                  </strong>
                  ；n(Ours)={luscKm.counts?.nOurs ?? '—'}，n(Others)={luscKm.counts?.nOthers ?? '—'}，重叠=
                  {luscKm.counts?.nOverlap ?? '—'}
                </Typography>
              </Box>
              <Box
                sx={(theme) => ({
                  width: '100%',
                  height: 380,
                  p: 1.25,
                  borderRadius: 1.5,
                  border: '1px solid',
                  borderColor: 'divider',
                  bgcolor: theme.palette.mode === 'dark' ? alpha(theme.palette.common.white, 0.02) : '#fbfcff',
                })}
              >
                <ResponsiveContainer width="100%" height="100%">
                  <LineChart data={luscKmChartData} margin={{ top: 8, right: 16, left: 8, bottom: 8 }}>
                    <CartesianGrid strokeDasharray="3 3" stroke={alpha('#90a4ae', 0.35)} />
                    <XAxis
                      dataKey="time"
                      type="number"
                      domain={['auto', 'auto']}
                      label={{ value: 'Time (months)', position: 'insideBottom', offset: -4 }}
                    />
                    <YAxis domain={[0, 1]} tickFormatter={(v) => v.toFixed(2)} width={48} label={{ value: 'Survival', angle: -90, position: 'insideLeft' }} />
                    <Tooltip
                      formatter={(value, name) => [typeof value === 'number' ? value.toFixed(4) : value, name]}
                      labelFormatter={(t) => `time ${typeof t === 'number' ? t.toFixed(4) : t}`}
                    />
                    <Legend />
                    {luscBaseline === 'others' ? (
                      <>
                        <Line
                          type="stepAfter"
                          dataKey="others"
                          name="Others (baseline)"
                          stroke="#37474F"
                          strokeWidth={2.5}
                          strokeDasharray="6 4"
                          dot={false}
                          isAnimationActive={false}
                        />
                        <Line
                          type="stepAfter"
                          dataKey="ours"
                          name="Ours"
                          stroke="#e53935"
                          strokeWidth={2.5}
                          dot={false}
                          isAnimationActive={false}
                        />
                      </>
                    ) : (
                      <>
                        <Line
                          type="stepAfter"
                          dataKey="ours"
                          name="Ours (baseline)"
                          stroke="#37474F"
                          strokeWidth={2.5}
                          strokeDasharray="6 4"
                          dot={false}
                          isAnimationActive={false}
                        />
                        <Line
                          type="stepAfter"
                          dataKey="others"
                          name="Others"
                          stroke="#1e88e5"
                          strokeWidth={2.5}
                          dot={false}
                          isAnimationActive={false}
                        />
                      </>
                    )}
                  </LineChart>
                </ResponsiveContainer>
              </Box>
              <Typography variant="caption" color="text.secondary" sx={{ display: 'block', mt: 1 }}>
                {luscKm.note || ''}
              </Typography>
            </>
          )}
        </CardContent>
      </Card>

    </Box>
  )
}

