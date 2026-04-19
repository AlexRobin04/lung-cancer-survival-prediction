import React, { useEffect, useMemo, useState } from 'react'
import {
  Alert,
  Box,
  Button,
  Card,
  CardContent,
  CardHeader,
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
import Toast from '../common/Toast.jsx'

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

  const [task, setTask] = useState(null)
  const [logText, setLogText] = useState('')
  const [history, setHistory] = useState([])
  const [queue, setQueue] = useState([])
  const [queueDeleting, setQueueDeleting] = useState(false)
  const [selectedTaskId, setSelectedTaskId] = useState('')

  const [historyDlgOpen, setHistoryDlgOpen] = useState(false)
  const [selectedDeleteIds, setSelectedDeleteIds] = useState([])
  const [deleteArtifacts, setDeleteArtifacts] = useState(true)

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
        setLogText(lg?.content || '')
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
        if (!stopped) setLogText(lg?.content || '')
      } catch {
        // ignore
      }
    }
    tick()
    const tmr = setInterval(tick, 5000)
    return () => {
      stopped = true
      clearInterval(tmr)
    }
  }, [task?.taskId])

  const start = async () => {
    setLoading(true)
    setError('')
    setNotice('')
    try {
      const payload = {
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
      }
      if (String(modelType) === 'EnsembleFeature') {
        payload.repeat = 1
        if (!freezeEnsembleBase) payload.finetuneEnsemble = true
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
                <FormControlLabel
                  sx={{ gridColumn: { xs: '1 / -1', md: 'span 2' }, alignSelf: 'center' }}
                  control={
                    <Checkbox checked={freezeEnsembleBase} onChange={(e) => setFreezeEnsembleBase(e.target.checked)} />
                  }
                  label="冻结五个基模型，仅训练融合头（推荐；取消勾选则端到端微调）"
                />
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
            选择历史任务后会自动拉取 status 与 log。
          </Typography>
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

