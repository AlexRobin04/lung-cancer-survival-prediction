import React, { useEffect, useMemo, useState, useCallback } from 'react'
import {
  Alert,
  Box,
  Button,
  Card,
  CardContent,
  CardHeader,
  Chip,
  CircularProgress,
  Dialog,
  DialogActions,
  DialogContent,
  DialogTitle,
  Divider,
  FormControl,
  IconButton,
  InputLabel,
  MenuItem,
  Select,
  Stack,
  Tooltip,
  TextField,
  Checkbox,
  FormControlLabel,
  Typography,
} from '@mui/material'
import { alpha } from '@mui/material/styles'
import DeleteOutlineIcon from '@mui/icons-material/DeleteOutline'
import { modelApi, trainingApi } from '../../services/api'
import useCancerOptions from '../../hooks/useCancerOptions'
import { CANCER_CN_MAP } from '../../constants/trainingOptions'
import { sanitizeTrainingLogContent } from '../../utils/trainingLogSanitize'
import Toast from '../common/Toast.jsx'

/** 与后端 ensemble_exclude / 文档 6.1 留一法一致 */
const ENSEMBLE_BRANCH_IDS = ['RRTMIL', 'AMIL', 'WiKG', 'DSMIL', 'S4MIL']

/** EnsembleFeature：前端一键提交的消融组合（共用当前表单中的癌种与超参） */
const ENSEMBLE_ABLATION_PRESETS = [
  {
    id: 'ab-gate-freeze',
    label: '门控 + 冻结五基线',
    fusion: 'gate',
    freezeBase: true,
    hint: 'fusion_mode=gate，仅训练对齐层 / 门控 / 融合头',
  },
  {
    id: 'ab-concat-freeze',
    label: '拼接 + 冻结五基线',
    fusion: 'concat',
    freezeBase: true,
    hint: 'fusion_mode=concat，与门控对照（同冻结策略）',
  },
  {
    id: 'ab-gate-ft',
    label: '门控 + 端到端微调',
    fusion: 'gate',
    freezeBase: false,
    hint: 'finetune_ensemble，显存与时间更高',
  },
  {
    id: 'ab-concat-ft',
    label: '拼接 + 端到端微调',
    fusion: 'concat',
    freezeBase: false,
    hint: '拼接 + 全模型微调，消融对照',
  },
]

/** 常用「用几路特征」子集，对应文档 6.1 思想（非穷举 2^5−1） */
const BRANCH_SUBSET_PRESETS = [
  {
    id: 'br-all',
    label: '五路全开',
    map: () => Object.fromEntries(ENSEMBLE_BRANCH_IDS.map((b) => [b, true])),
  },
  {
    id: 'br-rwd',
    label: '仅 RRT+WiKG+DSMIL',
    map: () => ({ RRTMIL: true, AMIL: false, WiKG: true, DSMIL: true, S4MIL: false }),
  },
  {
    id: 'br-mil3',
    label: '仅 AMIL+DSMIL+S4',
    map: () => ({ RRTMIL: false, AMIL: true, WiKG: false, DSMIL: true, S4MIL: true }),
  },
  {
    id: 'br-seq',
    label: '仅 WiKG+S4（图+序列）',
    map: () => ({ RRTMIL: false, AMIL: false, WiKG: true, DSMIL: false, S4MIL: true }),
  },
]

const sectionCardSx = (accent, { mb = 3 } = {}) => (theme) => ({
  mb,
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

export default function Training() {
  const [error, setError] = useState('')
  const [notice, setNotice] = useState('')
  const [loading, setLoading] = useState(false)

  const [models, setModels] = useState([])
  const { cancerOptions, cancer, setCancer } = useCancerOptions('LUSC')
  const [modelType, setModelType] = useState('RRTMIL')
  const [maxEpochs, setMaxEpochs] = useState(3)
  const [learningRate, setLearningRate] = useState(1e-5)
  const [repeat, setRepeat] = useState(1)
  const [seed, setSeed] = useState(1)
  const [kFolds, setKFolds] = useState(4)
  const [weightDecay, setWeightDecay] = useState(1e-5)
  const [earlyStopping, setEarlyStopping] = useState(false)
  const [freezeEnsembleBase, setFreezeEnsembleBase] = useState(true)
  /** EnsembleFeature：gate=门控加权融合；concat=对齐后拼接（消融） */
  const [ensembleFusion, setEnsembleFusion] = useState('gate')
  const [ablationChecks, setAblationChecks] = useState(() =>
    Object.fromEntries(ENSEMBLE_ABLATION_PRESETS.map((p) => [p.id, p.id === 'ab-gate-freeze' || p.id === 'ab-concat-freeze']))
  )
  /** 某路为 false 时，该基线在对齐后特征置零（等价于 ensembleExclude） */
  const [branchInclude, setBranchInclude] = useState(() =>
    Object.fromEntries(ENSEMBLE_BRANCH_IDS.map((b) => [b, true]))
  )

  const [task, setTask] = useState(null)
  const [logText, setLogText] = useState('')
  const [history, setHistory] = useState([])
  const [queue, setQueue] = useState([])
  const [queueDeleting, setQueueDeleting] = useState(false)
  const [selectedTaskId, setSelectedTaskId] = useState('')

  const [historyDlgOpen, setHistoryDlgOpen] = useState(false)
  const [selectedDeleteIds, setSelectedDeleteIds] = useState([])
  const [deleteArtifacts, setDeleteArtifacts] = useState(true)
  const [historyStopping, setHistoryStopping] = useState(false)

  const loadModels = async () => {
    try {
      const res = await modelApi.listModels()
      setModels(res?.models || [])
    } catch (e) {
      setModels([])
    }
  }

  useEffect(() => {
    loadModels()
  }, [])

  const formatCancerLabel = (code) => {
    const k = String(code || '').trim()
    if (!k) return ''
    const cn = CANCER_CN_MAP?.[k]
    return cn ? `${k}（${cn}）` : k
  }

  const loadHistory = async () => {
    try {
      const [h, q] = await Promise.all([trainingApi.history(), trainingApi.queue().catch(() => ({ queue: [] }))])
      setHistory(h?.tasks || h?.data?.tasks || [])
      setQueue(q?.queue || [])
    } catch {
      setHistory([])
      setQueue([])
    }
  }

  useEffect(() => {
    loadHistory()
  }, [])

  const removeQueueItem = async (taskId) => {
    if (!taskId || queueDeleting) return
    setQueueDeleting(true)
    setError('')
    try {
      const res = await trainingApi.deleteQueue({ taskIds: [taskId] })
      await loadHistory()
      if (res?.deletedTaskIds?.includes(selectedTaskId)) {
        setSelectedTaskId('')
        setTask(null)
        setLogText('')
      }
      setNotice(res?.deletedCount ? '已从队列移除' : '')
    } catch (e) {
      setError(e?.response?.data?.message || e.message || '移除队列失败')
    } finally {
      setQueueDeleting(false)
    }
  }

  const clearTrainingQueue = async () => {
    if (!queue.length || queueDeleting) return
    if (!window.confirm(`确定清空队列中的 ${queue.length} 个待执行任务？`)) return
    setQueueDeleting(true)
    setError('')
    try {
      const res = await trainingApi.deleteQueue({ deleteAll: true })
      await loadHistory()
      if (res?.deletedTaskIds?.includes(selectedTaskId)) {
        setSelectedTaskId('')
        setTask(null)
        setLogText('')
      }
      setNotice(res?.deletedCount ? `已清空 ${res.deletedCount} 项队列` : '')
    } catch (e) {
      setError(e?.response?.data?.message || e.message || '清空队列失败')
    } finally {
      setQueueDeleting(false)
    }
  }

  useEffect(() => {
    if (!selectedTaskId) return
    ;(async () => {
      try {
        const s = await trainingApi.status(selectedTaskId)
        setTask(s?.task || s)
        const lg = await trainingApi.log(selectedTaskId, 200)
        setLogText(sanitizeTrainingLogContent(lg?.content || ''))
      } catch (e) {
        setError(e?.response?.data?.message || e.message || '加载任务失败')
      }
    })()
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [selectedTaskId])

  useEffect(() => {
    if (!task?.taskId) return
    let stopped = false
    const tick = async () => {
      try {
        const s = await trainingApi.status(task.taskId)
        if (!stopped) setTask(s?.task || s)
        const lg = await trainingApi.log(task.taskId, 200)
        if (!stopped) setLogText(sanitizeTrainingLogContent(lg?.content || ''))
      } catch {
        // ignore
      }
    }
    tick()
    const running = String(task?.status || '').toLowerCase() === 'running'
    const ms = running ? 2000 : 4000
    const tmr = setInterval(tick, ms)
    return () => {
      stopped = true
      clearInterval(tmr)
    }
  }, [task?.taskId, task?.status])

  const buildBasePayload = useCallback(
    () => ({
      cancer,
      modelType,
      mode: 'transformer',
      maxEpochs: Number(maxEpochs),
      learningRate: Number(learningRate),
      kFolds: Math.min(20, Math.max(1, Number(kFolds) || 4)),
      weightDecay: Number(weightDecay) >= 0 ? Number(weightDecay) : 1e-5,
      earlyStopping,
      repeat: Number(repeat) || 1,
      seed: Number(seed) || 1,
    }),
    [cancer, modelType, maxEpochs, learningRate, kFolds, weightDecay, earlyStopping, repeat, seed]
  )

  const start = async () => {
    if (String(modelType) === 'EnsembleFeature') {
      const ex = ENSEMBLE_BRANCH_IDS.filter((b) => !branchInclude[b])
      if (ex.length >= ENSEMBLE_BRANCH_IDS.length) {
        setError('至少保留一路基线特征（文档 6.1：不可五路全关）')
        return
      }
    }
    setLoading(true)
    setError('')
    setNotice('')
    try {
      const payload = { ...buildBasePayload() }
      if (String(modelType) === 'EnsembleFeature') {
        payload.repeat = 1
        if (!freezeEnsembleBase) payload.finetuneEnsemble = true
        payload.ensembleFusion = ensembleFusion === 'concat' ? 'concat' : 'gate'
        const ex = ENSEMBLE_BRANCH_IDS.filter((b) => !branchInclude[b])
        if (ex.length > 0) payload.ensembleExclude = ex
      }
      const res = await trainingApi.start(payload)
      if (res?.taskId) {
        setTask({ taskId: res.taskId, status: res?.queued ? 'queued' : 'running' })
        setSelectedTaskId(res.taskId)
      }
      if (res?.queued) {
        setNotice('当前有任务在运行，已加入训练队列')
      } else {
        setNotice((Number(repeat) || 1) > 1 ? `已启动批量训练 ×${Number(repeat) || 1}` : '训练已启动')
      }
      await loadHistory()
    } catch (e) {
      setError(e?.response?.data?.message || e.message || '启动失败')
    } finally {
      setLoading(false)
    }
  }

  const submitEnsembleAblations = async () => {
    if (String(modelType) !== 'EnsembleFeature') return
    const ex = ENSEMBLE_BRANCH_IDS.filter((b) => !branchInclude[b])
    if (ex.length >= ENSEMBLE_BRANCH_IDS.length) {
      setError('至少保留一路基线特征')
      return
    }
    const selected = ENSEMBLE_ABLATION_PRESETS.filter((p) => ablationChecks[p.id])
    if (selected.length === 0) {
      setError('请至少勾选一项消融预设')
      return
    }
    setLoading(true)
    setError('')
    setNotice('')
    try {
      let lastRes = null
      for (const p of selected) {
        const payload = { ...buildBasePayload(), modelType: 'EnsembleFeature', repeat: 1 }
        payload.ensembleFusion = p.fusion === 'concat' ? 'concat' : 'gate'
        if (!p.freezeBase) payload.finetuneEnsemble = true
        if (ex.length > 0) payload.ensembleExclude = ex
        lastRes = await trainingApi.start(payload)
      }
      const n = selected.length
      const lastId = lastRes?.taskId
      if (lastId) {
        setTask({ taskId: lastId, status: lastRes?.queued ? 'queued' : 'running' })
        setSelectedTaskId(lastId)
      }
      setNotice(
        `已依次提交 ${n} 个 EnsembleFeature 消融任务（共用当前癌种与超参）。` +
          (n > 1 ? '单任务运行时后续项会自动入队；请在 Training Queue / History 中查看。' : '')
      )
      await loadHistory()
    } catch (e) {
      setError(e?.response?.data?.message || e.message || '消融提交失败')
    } finally {
      setLoading(false)
    }
  }

  /** 6.1 留一：依次提交 5 个任务，每次排除一路；共用当前融合方式、冻结选项与超参 */
  const submitLeaveOneOutBatch = async () => {
    if (String(modelType) !== 'EnsembleFeature') return
    setLoading(true)
    setError('')
    setNotice('')
    try {
      let lastRes = null
      for (const drop of ENSEMBLE_BRANCH_IDS) {
        const payload = { ...buildBasePayload(), modelType: 'EnsembleFeature', repeat: 1 }
        payload.ensembleFusion = ensembleFusion === 'concat' ? 'concat' : 'gate'
        if (!freezeEnsembleBase) payload.finetuneEnsemble = true
        payload.ensembleExclude = [drop]
        lastRes = await trainingApi.start(payload)
      }
      const lastId = lastRes?.taskId
      if (lastId) {
        setTask({ taskId: lastId, status: lastRes?.queued ? 'queued' : 'running' })
        setSelectedTaskId(lastId)
      }
      setNotice('已依次提交 5 个留一法（ensembleExclude 各一路）任务；与当前「融合方式」「冻结基线」设置一致。')
      await loadHistory()
    } catch (e) {
      setError(e?.response?.data?.message || e.message || '留一法提交失败')
    } finally {
      setLoading(false)
    }
  }

  const stop = async () => {
    if (!task?.taskId) return
    setLoading(true)
    setError('')
    setNotice('')
    try {
      await trainingApi.stop(task.taskId)
      setNotice('已发送停止请求')
      await loadHistory()
    } catch (e) {
      setError(e?.response?.data?.message || e.message || '停止失败')
    } finally {
      setLoading(false)
    }
  }

  const canStopSelectedHistory = useMemo(() => {
    const id = String(selectedTaskId || '').trim()
    if (!id || String(task?.taskId || '') !== id) return false
    const s = String(task?.status || '').toLowerCase()
    return s === 'running' || s === 'queued'
  }, [selectedTaskId, task])

  const stopSelectedFromHistory = async () => {
    if (!canStopSelectedHistory || !selectedTaskId) return
    setHistoryStopping(true)
    setError('')
    setNotice('')
    try {
      await trainingApi.stop(selectedTaskId)
      setNotice('已发送停止请求')
      await loadHistory()
      const s = await trainingApi.status(selectedTaskId)
      setTask(s?.task || s)
      const lg = await trainingApi.log(selectedTaskId, 200)
      setLogText(sanitizeTrainingLogContent(lg?.content || ''))
    } catch (e) {
      setError(e?.response?.data?.message || e.message || '停止失败')
    } finally {
      setHistoryStopping(false)
    }
  }

  const modelOptions = useMemo(() => models.map((m) => (typeof m === 'string' ? { id: m, name: m } : m)), [models])
  const allowedModelIds = useMemo(() => new Set(['AMIL', 'WiKG', 'DSMIL', 'S4MIL', 'RRTMIL', 'EnsembleFeature']), [])
  const filteredModelOptions = useMemo(() => modelOptions.filter((m) => allowedModelIds.has(String(m.id))), [modelOptions, allowedModelIds])

  const openHistoryManager = () => {
    setSelectedDeleteIds([])
    setDeleteArtifacts(true)
    setHistoryDlgOpen(true)
  }

  const toggleDeleteId = (id) => {
    setSelectedDeleteIds((prev) => {
      const s = new Set(prev || [])
      if (s.has(id)) s.delete(id)
      else s.add(id)
      return Array.from(s)
    })
  }

  const deleteSelected = async () => {
    if (selectedDeleteIds.length === 0) return
    const ok = window.confirm(`确认删除选中的 ${selectedDeleteIds.length} 条训练历史吗？`)
    if (!ok) return
    setLoading(true)
    setError('')
    setNotice('')
    try {
      const res = await trainingApi.deleteHistory({ taskIds: selectedDeleteIds, deleteArtifacts })
      setNotice(`已删除 ${res?.deletedCount ?? selectedDeleteIds.length} 条历史`)
      await loadHistory()
      setSelectedTaskId('')
      setTask(null)
      setLogText('')
      setSelectedDeleteIds([])
    } catch (e) {
      setError(e?.response?.data?.message || e.message || '删除失败')
    } finally {
      setLoading(false)
    }
  }

  const deleteAll = async () => {
    const ok = window.confirm('确认删除所有训练历史吗？')
    if (!ok) return
    setLoading(true)
    setError('')
    setNotice('')
    try {
      const res = await trainingApi.deleteHistory({ deleteAll: true, deleteArtifacts })
      setNotice(`已删除 ${res?.deletedCount ?? 0} 条历史`)
      await loadHistory()
      setSelectedTaskId('')
      setTask(null)
      setLogText('')
      setSelectedDeleteIds([])
    } catch (e) {
      setError(e?.response?.data?.message || e.message || '删除失败')
    } finally {
      setLoading(false)
    }
  }

  useEffect(() => {
    if (!allowedModelIds.has(String(modelType))) {
      // 后端模型列表可能包含更多实现；Training 页面仅保留指定模型
      setModelType('RRTMIL')
    }
  }, [modelType, allowedModelIds])

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
              ? 'linear-gradient(115deg, rgba(0,137,123,0.22) 0%, rgba(25,118,210,0.12) 100%)'
              : 'linear-gradient(115deg, rgba(0,137,123,0.12) 0%, rgba(25,118,210,0.06) 100%)',
        })}
      >
        <Typography variant="h4" gutterBottom sx={{ fontWeight: 700, mb: 1 }}>
          Training
        </Typography>
        <Typography variant="body2" color="text.secondary">
          配置癌种与模型、启动或停止训练，查看队列与当前任务日志。
        </Typography>
      </Box>

      {error && (
        <Alert severity="error" sx={{ mb: 2 }} onClose={() => setError('')}>
          {error}
        </Alert>
      )}
      <Toast open={!!notice} message={notice} severity="success" onClose={() => setNotice('')} />

      <Card sx={sectionCardSx('#1976d2')}>
        <CardHeader
          title="Start Training"
          titleTypographyProps={{ variant: 'subtitle1', fontWeight: 700 }}
        />
        <CardContent>
          <Box
            sx={{
              display: 'grid',
              gridTemplateColumns: { xs: '1fr', md: 'repeat(4, minmax(0, 1fr))' },
              gap: 2,
              alignItems: 'center',
              p: { xs: 0, sm: 0.5 },
            }}
          >
            {/* Row 1: Cancer / Model / maxEpochs / learningRate */}
            <FormControl sx={{ minWidth: 160 }}>
              <InputLabel id="cancer-label">Cancer</InputLabel>
              <Select labelId="cancer-label" label="Cancer" value={cancer} onChange={(e) => setCancer(e.target.value)}>
                {cancerOptions.map((opt) => (
                  <MenuItem key={opt.value} value={opt.value}>
                    {formatCancerLabel(opt.label || opt.value)}
                  </MenuItem>
                ))}
              </Select>
            </FormControl>
            <FormControl sx={{ minWidth: 220 }}>
              <InputLabel id="model-label">Model</InputLabel>
              <Select
                labelId="model-label"
                label="Model"
                value={modelType}
                onChange={(e) => setModelType(e.target.value)}
              >
                {filteredModelOptions.map((m) => (
                  <MenuItem key={m.id} value={m.id}>
                    {m.name || m.id}
                  </MenuItem>
                ))}
              </Select>
            </FormControl>
            <TextField
              label="maxEpochs"
              type="number"
              value={maxEpochs}
              onChange={(e) => setMaxEpochs(e.target.value)}
              sx={{ minWidth: 140 }}
            />
            <TextField
              label="learningRate"
              type="number"
              value={learningRate}
              onChange={(e) => setLearningRate(e.target.value)}
              sx={{ minWidth: 180 }}
              inputProps={{ step: '0.000001' }}
            />

            {/* Row 2: kFolds / weightDecay / repeat / seed */}
            <TextField
              label="kFolds"
              type="number"
              value={kFolds}
              onChange={(e) => setKFolds(e.target.value)}
              sx={{ minWidth: 120 }}
              inputProps={{ min: 1, max: 20, step: 1 }}
              helperText="交叉验证折数（--k）"
            />
            <TextField
              label="weightDecay"
              type="number"
              value={weightDecay}
              onChange={(e) => setWeightDecay(e.target.value)}
              sx={{ minWidth: 160 }}
              inputProps={{ step: '0.0000001', min: 0 }}
              helperText="L2 权重衰减（--reg）"
            />
            <TextField
              label="repeat"
              type="number"
              value={repeat}
              onChange={(e) => setRepeat(e.target.value)}
              sx={{ minWidth: 140 }}
              inputProps={{ min: 1, max: 200, step: 1 }}
              helperText="重复训练次数"
            />
            <TextField
              label="seed"
              type="number"
              value={seed}
              onChange={(e) => setSeed(e.target.value)}
              sx={{ minWidth: 180 }}
              inputProps={{ min: 0, max: 10000000, step: 1 }}
              helperText="baseSeed（repeat 时递增）"
            />

            {/* Row 3: early stop + actions */}
            <FormControlLabel
              sx={{ gridColumn: { xs: '1 / -1', md: 'span 2' }, alignSelf: 'center' }}
              control={
                <Checkbox checked={earlyStopping} onChange={(e) => setEarlyStopping(e.target.checked)} />
              }
              label="早停（--early_stopping，按验证集 val_error）"
            />
            {String(modelType) === 'EnsembleFeature' && (
              <>
                <Alert severity="info" sx={{ gridColumn: { xs: '1 / -1', md: 'span 2' } }}>
                  EnsembleFeature 为特征级集成。后端会在每个训练折上，自动从{' '}
                  <code>uploaded_features/best_models.json</code> 与 <code>tasks.json</code> 解析五个基模型（RRTMIL/AMIL/WiKG/DSMIL/S4MIL）当前癌种
                  + mode 的最佳任务，并加载对应 <code>resultsDir/s_折_checkpoint.pt</code>。请先在平台各训好五个基模型并产生 best
                  记录；若需用手动权重目录，请用命令行 <code>--ensemble_ckpt_dir</code> 启动训练。
                </Alert>
                <Box sx={{ gridColumn: '1 / -1' }}>
                  <Typography variant="subtitle2" sx={{ fontWeight: 600, mb: 0.5 }}>
                    基线特征子集（用几路特征 · 文档 6.1）
                  </Typography>
                  <Typography variant="caption" color="text.secondary" sx={{ display: 'block', mb: 1 }}>
                    勾选 = 该路参与融合；取消勾选 = 对齐后该路置零，等价于请求体{' '}
                    <code>ensembleExclude</code>。单次 Start、下方「融合×冻结」批量消融都会带上当前子集。留一法入队与这里独立，固定每次只关一路。
                  </Typography>
                  <Stack direction="row" flexWrap="wrap" gap={1} sx={{ mb: 1 }} alignItems="center">
                    {ENSEMBLE_BRANCH_IDS.map((b) => (
                      <FormControlLabel
                        key={b}
                        sx={{ mr: 1 }}
                        control={
                          <Checkbox
                            size="small"
                            checked={!!branchInclude[b]}
                            onChange={() => setBranchInclude((prev) => ({ ...prev, [b]: !prev[b] }))}
                          />
                        }
                        label={<Typography variant="body2">{b}</Typography>}
                      />
                    ))}
                  </Stack>
                  <Typography variant="caption" color="text.secondary" sx={{ display: 'block', mb: 0.5 }}>
                    快捷子集
                  </Typography>
                  <Stack direction="row" flexWrap="wrap" gap={0.75} sx={{ mb: 1 }}>
                    {BRANCH_SUBSET_PRESETS.map((sp) => (
                      <Chip
                        key={sp.id}
                        size="small"
                        label={sp.label}
                        variant="outlined"
                        onClick={() => setBranchInclude(sp.map())}
                      />
                    ))}
                    {ENSEMBLE_BRANCH_IDS.map((b) => (
                      <Chip
                        key={`solo-off-${b}`}
                        size="small"
                        label={`只关 ${b}`}
                        variant="outlined"
                        onClick={() =>
                          setBranchInclude(
                            Object.fromEntries(ENSEMBLE_BRANCH_IDS.map((x) => [x, x !== b]))
                          )
                        }
                      />
                    ))}
                  </Stack>
                </Box>
                <FormControlLabel
                  sx={{ gridColumn: { xs: '1 / -1', md: 'span 2' }, alignSelf: 'center' }}
                  control={
                    <Checkbox checked={freezeEnsembleBase} onChange={(e) => setFreezeEnsembleBase(e.target.checked)} />
                  }
                  label="冻结五个基模型，仅训练融合头（推荐；取消勾选则端到端微调）"
                />
                <FormControl size="small" sx={{ gridColumn: { xs: '1 / -1', md: 'span 2' }, minWidth: 280 }}>
                  <InputLabel id="ensemble-fusion-label">融合方式</InputLabel>
                  <Select
                    labelId="ensemble-fusion-label"
                    label="融合方式"
                    value={ensembleFusion}
                    onChange={(e) => setEnsembleFusion(e.target.value)}
                  >
                    <MenuItem value="gate">门控加权（推荐）</MenuItem>
                    <MenuItem value="concat">对齐后拼接（消融对照）</MenuItem>
                  </Select>
                </FormControl>
                <Divider sx={{ gridColumn: '1 / -1' }} />
                <Box sx={{ gridColumn: '1 / -1' }}>
                  <Typography variant="subtitle2" sx={{ fontWeight: 600, mb: 0.5 }}>
                    消融实验（批量入队）
                  </Typography>
                  <Typography variant="caption" color="text.secondary" sx={{ display: 'block', mb: 1 }}>
                    与上方单次 Start 共用：Cancer、maxEpochs、learningRate、kFolds、weightDecay、早停、seed，以及当前{' '}
                    <strong>基线特征子集</strong>（<code>ensembleExclude</code>）；每项为独立任务，按列表顺序提交（有运行中任务时自动排队）。
                  </Typography>
                  <Stack spacing={0.5} sx={{ mb: 1 }}>
                    {ENSEMBLE_ABLATION_PRESETS.map((p) => (
                      <FormControlLabel
                        key={p.id}
                        sx={{ alignItems: 'flex-start', ml: 0 }}
                        control={
                          <Checkbox
                            size="small"
                            checked={!!ablationChecks[p.id]}
                            onChange={() => setAblationChecks((prev) => ({ ...prev, [p.id]: !prev[p.id] }))}
                          />
                        }
                        label={
                          <Box>
                            <Typography variant="body2">{p.label}</Typography>
                            <Typography variant="caption" color="text.secondary">
                              {p.hint}
                            </Typography>
                          </Box>
                        }
                      />
                    ))}
                  </Stack>
                  <Stack direction="row" flexWrap="wrap" gap={1} alignItems="center">
                    <Button
                      variant="outlined"
                      size="small"
                      disabled={loading}
                      onClick={() =>
                        setAblationChecks(Object.fromEntries(ENSEMBLE_ABLATION_PRESETS.map((x) => [x.id, true])))
                      }
                    >
                      全选
                    </Button>
                    <Button
                      variant="outlined"
                      size="small"
                      disabled={loading}
                      onClick={() =>
                        setAblationChecks(Object.fromEntries(ENSEMBLE_ABLATION_PRESETS.map((x) => [x.id, false])))
                      }
                    >
                      全不选
                    </Button>
                    <Button
                      variant="contained"
                      color="secondary"
                      size="small"
                      disabled={
                        loading ||
                        !ENSEMBLE_ABLATION_PRESETS.some((p) => ablationChecks[p.id])
                      }
                      onClick={submitEnsembleAblations}
                    >
                      提交所选消融（
                      {ENSEMBLE_ABLATION_PRESETS.filter((p) => ablationChecks[p.id]).length} 项）
                    </Button>
                    <Button
                      variant="outlined"
                      color="secondary"
                      size="small"
                      disabled={loading}
                      onClick={submitLeaveOneOutBatch}
                    >
                      留一法入队（5 项）
                    </Button>
                  </Stack>
                </Box>
              </>
            )}
            <Button variant="contained" onClick={start} disabled={loading} sx={{ height: 40 }}>
              {loading ? <CircularProgress size={20} color="inherit" /> : 'Start'}
            </Button>
            <Button
              variant="outlined"
              color="warning"
              onClick={stop}
              disabled={loading || !task?.taskId}
              sx={{ height: 40 }}
            >
              Stop
            </Button>
          </Box>
        </CardContent>
      </Card>

      <Card sx={sectionCardSx('#ed6c02')}>
        <CardHeader
          title="Training Queue"
          titleTypographyProps={{ variant: 'subtitle1', fontWeight: 700 }}
          action={
            queue.length > 0 ? (
              <Button
                size="small"
                color="error"
                variant="outlined"
                disabled={queueDeleting}
                onClick={clearTrainingQueue}
              >
                清空队列
              </Button>
            ) : null
          }
        />
        <CardContent>
          {queue.length === 0 ? (
            <Typography variant="body2" color="text.secondary">
              当前队列为空（有运行任务时，新任务会自动入队并在前序结束后执行）。
            </Typography>
          ) : (
            <Box sx={{ display: 'grid', gap: 1 }}>
              {queue.map((q, idx) => (
                <Box
                  key={q.taskId}
                  sx={(theme) => ({
                    display: 'flex',
                    alignItems: 'center',
                    gap: 1,
                    py: 1,
                    px: 1.25,
                    pr: 0.5,
                    borderRadius: 1.25,
                    border: '1px solid',
                    borderColor: alpha('#ed6c02', theme.palette.mode === 'dark' ? 0.35 : 0.28),
                    bgcolor: alpha('#ed6c02', theme.palette.mode === 'dark' ? 0.1 : 0.04),
                  })}
                >
                  <Typography variant="body2" sx={{ flex: 1, minWidth: 0 }}>
                    #{idx + 1} {q.modelType} — {q.cancer} — k:{q.kFolds ?? '—'} — epochs:{q.maxEpochs} — wd:
                    {q.weightDecay ?? '—'} — 早停:{q.earlyStopping ? '是' : '否'} — repeat:{q.repeatTotal ?? 1} — seed:
                    {q.seed ?? '—'}
                    {q.finetuneEnsemble ? ' — finetune' : ''} — queuedAt:{q.queuedAt || '—'}
                  </Typography>
                  <Tooltip title="从队列移除（仅排队任务）">
                    <span>
                      <IconButton
                        size="small"
                        color="error"
                        aria-label="从队列移除"
                        disabled={queueDeleting}
                        onClick={() => removeQueueItem(q.taskId)}
                      >
                        <DeleteOutlineIcon fontSize="small" />
                      </IconButton>
                    </span>
                  </Tooltip>
                </Box>
              ))}
            </Box>
          )}
        </CardContent>
      </Card>

      <Card sx={sectionCardSx('#7b1fa2')}>
        <CardHeader
          title="Recent Training Tasks"
          titleTypographyProps={{ variant: 'subtitle1', fontWeight: 700 }}
          action={
            <Stack direction="row" spacing={1}>
              <Button variant="outlined" onClick={loadHistory}>
                Refresh
              </Button>
              <Button variant="outlined" color="error" onClick={openHistoryManager}>
                管理历史
              </Button>
            </Stack>
          }
        />
        <CardContent>
          <FormControl
            fullWidth
            sx={(theme) => ({
              '& .MuiOutlinedInput-root': {
                bgcolor: theme.palette.mode === 'dark' ? 'action.hover' : 'background.paper',
              },
            })}
          >
            <InputLabel id="hist-label">Task</InputLabel>
            <Select
              labelId="hist-label"
              label="Task"
              value={selectedTaskId}
              onChange={(e) => setSelectedTaskId(e.target.value)}
            >
              {history.map((t) => (
                <MenuItem key={t.taskId} value={t.taskId}>
                  {t.modelType} — {t.status}{t.isBestForModel ? ' — Best' : ''} — {t.startedAt || ''}
                </MenuItem>
              ))}
            </Select>
          </FormControl>
          <Typography variant="caption" color="text.secondary" sx={{ mt: 1, display: 'block' }}>
            选择历史任务后会自动拉取 status 与 log。下方橙色按钮可停止<strong>当前下拉框选中</strong>的任务（仅 running / queued）。
          </Typography>
          <Divider sx={{ my: 2 }} />
          <Button
            fullWidth
            variant={canStopSelectedHistory ? 'contained' : 'outlined'}
            color="warning"
            size="medium"
            disabled={!canStopSelectedHistory || historyStopping || loading}
            onClick={stopSelectedFromHistory}
            sx={{ fontWeight: 700 }}
          >
            {historyStopping ? '正在停止…' : '停止所选任务（历史列表）'}
          </Button>
          {!canStopSelectedHistory && selectedTaskId && (
            <Typography variant="caption" color="text.secondary" sx={{ mt: 1, display: 'block' }}>
              当前选中任务状态为「{String(task?.status || '未知')}」，已结束的任务无法停止；请换选 running 或 queued 条目。
            </Typography>
          )}
        </CardContent>
      </Card>

      <Dialog open={historyDlgOpen} onClose={() => setHistoryDlgOpen(false)} fullWidth maxWidth="md">
        <DialogTitle>删除训练历史</DialogTitle>
        <DialogContent dividers>
          <Typography variant="body2" color="text.secondary" sx={{ mb: 1.5 }}>
            支持勾选删除或一键全删。运行中的任务不会被删除。
          </Typography>
          <FormControlLabel
            control={<Checkbox checked={deleteArtifacts} onChange={(e) => setDeleteArtifacts(e.target.checked)} />}
            label="同时删除训练日志与结果目录（resultsDir / logPath）"
          />
          <Divider sx={{ my: 1.5 }} />
          <Box sx={{ maxHeight: 360, overflow: 'auto' }}>
            {(history || []).map((t) => {
              const id = t.taskId
              const disabled = String(t.status || '').toLowerCase() === 'running'
              const checked = selectedDeleteIds.includes(id)
              return (
                <Box
                  key={id}
                  sx={{
                    display: 'flex',
                    alignItems: 'center',
                    justifyContent: 'space-between',
                    gap: 1.5,
                    py: 0.5,
                  }}
                >
                  <FormControlLabel
                    sx={{ m: 0, flex: 1 }}
                    control={<Checkbox checked={checked} disabled={disabled} onChange={() => toggleDeleteId(id)} />}
                    label={
                      <Typography variant="body2" sx={{ wordBreak: 'break-all' }}>
                        {t.modelType} — {t.cancer} — {t.status}
                        {t.isBestForModel ? ' — Best' : ''} — {t.startedAt || ''}
                      </Typography>
                    }
                  />
                  <Typography variant="caption" color="text.secondary" sx={{ minWidth: 92, textAlign: 'right' }}>
                    {disabled ? '运行中' : ''}
                  </Typography>
                </Box>
              )
            })}
          </Box>
        </DialogContent>
        <DialogActions sx={{ display: 'flex', justifyContent: 'space-between', gap: 1, flexWrap: 'wrap' }}>
          <Button color="error" onClick={deleteAll} disabled={loading || (history || []).length === 0}>
            一键删除全部
          </Button>
          <Box sx={{ display: 'flex', gap: 1 }}>
            <Button onClick={() => setHistoryDlgOpen(false)}>取消</Button>
            <Button variant="contained" color="error" onClick={deleteSelected} disabled={loading || selectedDeleteIds.length === 0}>
              删除所选（{selectedDeleteIds.length}）
            </Button>
          </Box>
        </DialogActions>
      </Dialog>

      <Card sx={sectionCardSx('#00897b', { mb: 0 })}>
        <CardHeader title="Current Task" titleTypographyProps={{ variant: 'subtitle1', fontWeight: 700 }} />
        <CardContent>
          {!task ? (
            <Typography variant="body2" color="text.secondary">
              暂无任务
            </Typography>
          ) : (
            <>
              <Box
                sx={(theme) => ({
                  p: 1.5,
                  mb: 2,
                  borderRadius: 1.5,
                  border: '1px solid',
                  borderColor: 'divider',
                  bgcolor: theme.palette.mode === 'dark' ? 'action.hover' : alpha('#00897b', 0.06),
                })}
              >
                <Typography variant="body2">
                  taskId: <code>{task.taskId}</code>
                </Typography>
                <Typography variant="body2" sx={{ mt: 0.5 }}>
                  status: {task.status}；progress: {task.progress ?? '—'}%；epoch: {task.epoch ?? '—'}；cIndex:{' '}
                  {task.cIndex ?? '—'}
                </Typography>
                {String(task.modelType || '') === 'EnsembleFeature' && (
                  <Typography variant="body2" color="text.secondary" sx={{ mt: 0.5 }}>
                    EnsembleFeature · 基线权重由后端按 best_models/tasks 自动解析（或命令行 --ensemble_ckpt_dir）
                    {task.finetuneEnsemble ? ' · 端到端微调' : ' · 冻结基模型（默认）'}
                  </Typography>
                )}
                {task.seed != null && (
                  <Typography variant="body2" color="text.secondary" sx={{ mt: 0.5 }}>
                    seed: {task.seed}
                  </Typography>
                )}
              </Box>
              <Divider sx={{ my: 2 }} />
              <Typography variant="subtitle2" color="text.secondary" sx={{ mb: 1 }}>
                Log (tail)
              </Typography>
              <Box
                sx={{
                  whiteSpace: 'pre-wrap',
                  fontFamily: 'ui-monospace, SFMono-Regular, Menlo, Monaco, Consolas, monospace',
                  fontSize: 12,
                  background: '#0b1020',
                  color: '#d7e3ff',
                  borderRadius: 1,
                  p: 2,
                  maxHeight: 380,
                  overflow: 'auto',
                }}
              >
                {logText || '—'}
              </Box>
            </>
          )}
        </CardContent>
      </Card>
    </Box>
  )
}

