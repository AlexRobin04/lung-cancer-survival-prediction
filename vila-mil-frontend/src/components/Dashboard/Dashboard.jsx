import React, { useEffect, useMemo, useState } from 'react'
import {
  Alert,
  Avatar,
  Box,
  Card,
  CardContent,
  CardHeader,
  Grid,
  Typography,
  Button,
  List,
  ListItemButton,
  ListItemText,
  Divider,
  FormControl,
  InputLabel,
  MenuItem,
  Select,
  Dialog,
  DialogTitle,
  DialogContent,
  DialogActions,
  Chip,
  CircularProgress,
} from '@mui/material'
import { alpha } from '@mui/material/styles'
import InsightsOutlinedIcon from '@mui/icons-material/InsightsOutlined'
import StorageOutlinedIcon from '@mui/icons-material/StorageOutlined'
import TrendingUpOutlinedIcon from '@mui/icons-material/TrendingUpOutlined'
import ModelTrainingOutlinedIcon from '@mui/icons-material/ModelTrainingOutlined'
import AssessmentOutlinedIcon from '@mui/icons-material/AssessmentOutlined'
import StopOutlinedIcon from '@mui/icons-material/StopOutlined'
import { healthApi, trainingApi, evaluationApi, predictApi, dataApi, clinicalApi } from '../../services/api'
import { sanitizeTrainingLogContent } from '../../utils/trainingLogSanitize'

const nowSec = () => Date.now() / 1000
const parseToSec = (t) => {
  const n = Number(t)
  if (Number.isFinite(n) && n > 0) return n
  if (!t) return null
  const ms = Date.parse(String(t))
  return Number.isFinite(ms) ? ms / 1000 : null
}

const fmtPct = (v) => {
  const n = Number(v)
  return Number.isFinite(n) ? `${n.toFixed(2)}%` : '—'
}

/** 从启动命令中解析 `--flag` 后的第一个 token */
const parseCmdArg = (cmd, flag) => {
  const esc = flag.replace(/[.*+?^${}()|[\]\\]/g, '\\$&')
  const m = String(cmd || '').match(new RegExp(`${esc}\\s+(\\S+)`, 'i'))
  return m ? m[1].trim() : ''
}

/**
 * 合并 tasks.json 字段与 command 字符串，供任务详情展示「本次参数 / 消融」。
 */
const buildTaskConfigSnapshot = (task) => {
  if (!task) return null
  const cmd = String(task.command || '')
  const model = String(task.modelType || '')
  const early =
    typeof task.earlyStopping === 'boolean'
      ? task.earlyStopping
      : /\b--early_stopping\b/i.test(cmd)
  const fusionRaw = task.ensembleFusion || parseCmdArg(cmd, '--ensemble_fusion')
  const fusion = fusionRaw ? String(fusionRaw).toLowerCase() : model === 'EnsembleFeature' ? 'gate' : ''
  let exclude = []
  if (Array.isArray(task.ensembleExclude)) exclude = [...task.ensembleExclude]
  else if (task.ensembleExclude && typeof task.ensembleExclude === 'string') {
    exclude = task.ensembleExclude.split(/[,;]/).map((s) => s.trim()).filter(Boolean)
  }
  if (exclude.length === 0 && /--ensemble_exclude\b/i.test(cmd)) {
    exclude = parseCmdArg(cmd, '--ensemble_exclude')
      .split(/[,;]/)
      .map((s) => s.trim())
      .filter(Boolean)
  }
  const finetune = Boolean(task.finetuneEnsemble) || /\b--finetune_ensemble\b/i.test(cmd)
  const wd = task.weightDecay != null && task.weightDecay !== '' ? task.weightDecay : parseCmdArg(cmd, '--reg')
  const k = task.kFolds != null ? task.kFolds : parseCmdArg(cmd, '--k')
  const me = task.maxEpochs != null ? task.maxEpochs : parseCmdArg(cmd, '--max_epochs')
  const lr = task.learningRate != null ? task.learningRate : parseCmdArg(cmd, '--lr')
  const seed = task.seed != null ? task.seed : parseCmdArg(cmd, '--seed')
  const mode = task.mode || parseCmdArg(cmd, '--mode')
  const ckptDir = (task.ensembleCkptDir || '').trim() || parseCmdArg(cmd, '--ensemble_ckpt_dir') || ''

  return {
    cancer: task.cancer ?? '—',
    mode: mode || '—',
    model,
    maxEpochs: me ?? '—',
    learningRate: lr ?? '—',
    kFolds: k ?? '—',
    weightDecay: wd !== '' && wd != null ? wd : '—',
    seed: seed ?? '—',
    earlyStopping: early,
    batchSize: task.batchSize,
    repeatTotal: task.repeatTotal ?? task.repeat,
    ensembleFusion: fusion,
    finetuneEnsemble: finetune,
    ensembleExclude: exclude,
    ensembleCkptDir: ckptDir,
  }
}

const trainingTaskCanStop = (t) => {
  const s = String(t?.status || '').toLowerCase()
  return s === 'running' || s === 'queued'
}

/** 分区卡片：顶边色带 + 浅色底 + 图标，便于一眼区分各模块 */
function DashboardPanel({ accent, icon, title, subheader, action, children }) {
  return (
    <Card
      elevation={0}
      sx={(theme) => ({
        height: '100%',
        borderRadius: 2,
        overflow: 'hidden',
        border: '1px solid',
        borderColor: 'divider',
        borderTop: '4px solid',
        borderTopColor: accent,
        bgcolor: alpha(accent, theme.palette.mode === 'dark' ? 0.14 : 0.06),
        boxShadow: `0 2px 14px ${alpha(accent, 0.12)}`,
        transition: 'box-shadow 0.2s ease, transform 0.2s ease',
        '&:hover': {
          boxShadow: `0 6px 20px ${alpha(accent, 0.18)}`,
        },
      })}
    >
      <CardHeader
        avatar={
          <Avatar
            variant="rounded"
            sx={{
              width: 40,
              height: 40,
              bgcolor: (theme) => alpha(accent, theme.palette.mode === 'dark' ? 0.35 : 0.2),
              color: accent,
            }}
          >
            {icon}
          </Avatar>
        }
        title={
          <Typography variant="subtitle1" sx={{ fontWeight: 700 }}>
            {title}
          </Typography>
        }
        subheader={subheader}
        action={action}
        sx={{
          pb: 0,
          '& .MuiCardHeader-subheader': { mt: 0.5 },
        }}
      />
      <CardContent sx={{ pt: 1.5 }}>{children}</CardContent>
    </Card>
  )
}

export default function Dashboard() {
  const [error, setError] = useState('')
  const [health, setHealth] = useState(null)
  const [tasks, setTasks] = useState([])
  const [runs, setRuns] = useState([])
  const [preds, setPreds] = useState([])
  const [datasets, setDatasets] = useState(null)
  const [cases, setCases] = useState([])
  const [selectedTaskId, setSelectedTaskId] = useState('')
  const [selectedTask, setSelectedTask] = useState(null)
  const [selectedLog, setSelectedLog] = useState('')
  const [selectedRunId, setSelectedRunId] = useState('')
  const [selectedRunCurves, setSelectedRunCurves] = useState(null)
  const [runDetailLoading, setRunDetailLoading] = useState(false)
  /** 默认「全部」避免只显示最近 15 分钟时列表为空，看不到任务与停止入口 */
  const [timeRangeMin, setTimeRangeMin] = useState(0)
  const [detailOpen, setDetailOpen] = useState(false)
  const [runDetailOpen, setRunDetailOpen] = useState(false)
  const [stoppingTaskId, setStoppingTaskId] = useState('')

  const load = async () => {
    setError('')
    try {
      const [h, th, r, p, d, c] = await Promise.all([
        healthApi.health(),
        trainingApi.history(),
        evaluationApi.runs(),
        predictApi.listPredictions(20),
        dataApi.getDatasets(),
        clinicalApi.listCases(),
      ])
      setHealth(h)
      setTasks(th?.tasks || th?.data?.tasks || [])
      setRuns(r?.runs || [])
      setPreds(p?.items || [])
      setDatasets(d)
      setCases(c?.cases || [])
    } catch (e) {
      setError(e?.response?.data?.message || e.message || '加载失败')
    }
  }

  useEffect(() => {
    load()
  }, [])

  const filteredTasks = useMemo(() => {
    const ts = tasks || []
    if (timeRangeMin === 0) return ts
    const cutoff = nowSec() - timeRangeMin * 60
    return ts.filter((t) => {
      const s = parseToSec(t.startedAtTs) ?? parseToSec(t.startedAt)
      return s != null && s >= cutoff
    })
  }, [tasks, timeRangeMin])

  const recentTasks = useMemo(() => filteredTasks.slice(0, 15), [filteredTasks])
  const completedRuns = useMemo(() => (runs || []).filter((x) => x?.hasMetrics).slice(0, 15), [runs])
  const selectedRunMeta = useMemo(
    () => (runs || []).find((r) => r.taskId === selectedRunId) || null,
    [runs, selectedRunId]
  )

  const loadTaskDetail = async (taskId) => {
    if (!taskId) return
    try {
      const s = await trainingApi.status(taskId)
      setSelectedTask(s?.task || s)
      const lg = await trainingApi.log(taskId, 2000)
      setSelectedLog(sanitizeTrainingLogContent(lg?.content || ''))
    } catch (e) {
      setError(e?.response?.data?.message || e.message || '加载任务详情失败')
    }
  }

  const stopTrainingTask = async (taskId) => {
    if (!taskId || stoppingTaskId) return
    setStoppingTaskId(taskId)
    setError('')
    try {
      await trainingApi.stop(taskId)
      await load()
      if (selectedTaskId === taskId) {
        await loadTaskDetail(taskId)
      }
    } catch (e) {
      setError(e?.response?.data?.message || e.message || '停止任务失败')
    } finally {
      setStoppingTaskId('')
    }
  }

  const loadRunDetail = async (taskId) => {
    if (!taskId) return
    setRunDetailLoading(true)
    try {
      const c = await evaluationApi.curves(taskId)
      setSelectedRunCurves(c)
    } catch (e) {
      setError(e?.response?.data?.message || e.message || '加载评估详情失败')
      setSelectedRunCurves(null)
    } finally {
      setRunDetailLoading(false)
    }
  }

  useEffect(() => {
    if (!selectedTaskId) return
    let stopped = false
    const tick = async () => {
      if (stopped) return
      await loadTaskDetail(selectedTaskId)
    }
    tick()
    const ms = detailOpen ? 2000 : 4000
    const tmr = setInterval(tick, ms)
    return () => {
      stopped = true
      clearInterval(tmr)
    }
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [selectedTaskId, detailOpen])

  useEffect(() => {
    let stopped = false
    const tick = async () => {
      if (stopped) return
      await load()
    }
    tick()
    const tmr = setInterval(tick, 5000)
    return () => {
      stopped = true
      clearInterval(tmr)
    }
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [timeRangeMin])

  return (
    <Box sx={{ mt: 2 }}>
      <Box
        sx={{
          display: 'flex',
          alignItems: 'center',
          justifyContent: 'space-between',
          mb: 2,
          p: 2,
          borderRadius: 2,
          background: (theme) =>
            theme.palette.mode === 'dark'
              ? `linear-gradient(110deg, ${alpha(theme.palette.primary.main, 0.2)} 0%, ${alpha(theme.palette.primary.dark, 0.08)} 100%)`
              : `linear-gradient(110deg, ${alpha(theme.palette.primary.main, 0.12)} 0%, ${alpha(theme.palette.primary.light, 0.06)} 100%)`,
          border: '1px solid',
          borderColor: 'divider',
        }}
      >
        <Box>
          <Typography variant="h4" sx={{ fontWeight: 700 }}>
            Dashboard
          </Typography>
          <Typography variant="body2" color="text.secondary" sx={{ mt: 0.5 }}>
            总览服务状态、数据规模、近期训练与评估
          </Typography>
        </Box>
        <Button variant="contained" onClick={load} sx={{ flexShrink: 0 }}>
          Refresh
        </Button>
      </Box>

      {error && (
        <Alert severity="error" sx={{ mb: 2 }} onClose={() => setError('')}>
          {error}
        </Alert>
      )}

      <Grid container spacing={3}>
        <Grid item xs={12} md={4}>
          <DashboardPanel
            accent="#1565c0"
            icon={<InsightsOutlinedIcon fontSize="small" />}
            title="Project Overview"
            action={
              <Button variant="text" size="small" onClick={load} sx={{ color: '#1565c0' }}>
                REFRESH
              </Button>
            }
          >
            <Typography variant="body2" color="text.secondary" sx={{ lineHeight: 1.7 }}>
              本系统用于“基于病理图像/特征的肺癌生存风险预测”，支持特征上传与病例管理、模型训练与评估、风险预测与分层展示。
            </Typography>
            <Divider sx={{ my: 2 }} />
            <Typography variant="body2">
              <Box component="span" color="text.secondary">
                Service:{' '}
              </Box>
              <Box component="span" sx={{ fontWeight: 600, color: health?.ok ? 'success.main' : 'text.primary' }}>
                {health?.ok ? 'OK' : '—'}
              </Box>
            </Typography>
            <Typography variant="body2" sx={{ mt: 0.5 }}>
              <Box component="span" color="text.secondary">
                Total Tasks:{' '}
              </Box>
              <Box component="span" sx={{ fontWeight: 600 }}>
                {tasks.length}
              </Box>
            </Typography>
            <Typography variant="body2" sx={{ mt: 0.5 }}>
              <Box component="span" color="text.secondary">
                Running:{' '}
              </Box>
              <Box component="span" sx={{ fontWeight: 600, color: 'warning.main' }}>
                {(tasks || []).filter((t) => t.status === 'running').length}
              </Box>
              <Box component="span" color="text.secondary" sx={{ mx: 1 }}>
                ·
              </Box>
              <Box component="span" color="text.secondary">
                Completed:{' '}
              </Box>
              <Box component="span" sx={{ fontWeight: 600, color: 'success.main' }}>
                {(tasks || []).filter((t) => t.status === 'completed').length}
              </Box>
            </Typography>
          </DashboardPanel>
        </Grid>

        <Grid item xs={12} md={4}>
          <DashboardPanel accent="#2e7d32" icon={<StorageOutlinedIcon fontSize="small" />} title="Data Summary">
            <Typography variant="body2" color="text.secondary">
              特征登记总数
            </Typography>
            <Typography variant="h4" sx={{ mt: 0.5, fontWeight: 800, color: '#2e7d32' }}>
              {datasets?.totalFiles ?? '—'}
            </Typography>
            <Box
              sx={{
                mt: 2,
                p: 1.5,
                borderRadius: 1,
                bgcolor: (theme) => alpha('#2e7d32', theme.palette.mode === 'dark' ? 0.2 : 0.08),
              }}
            >
              <Typography variant="body2" color="text.secondary">
                病例数
              </Typography>
              <Typography variant="h6" sx={{ fontWeight: 700 }}>
                {cases.length}
              </Typography>
            </Box>
          </DashboardPanel>
        </Grid>

        <Grid item xs={12} md={4}>
          <DashboardPanel accent="#ed6c02" icon={<TrendingUpOutlinedIcon fontSize="small" />} title="Predictions (recent)">
            <Typography variant="body2" color="text.secondary">
              最近三条
            </Typography>
            {(preds || []).slice(0, 3).map((it) => (
              <Box
                key={it.id}
                sx={{
                  mt: 1.25,
                  p: 1.25,
                  borderRadius: 1,
                  border: '1px solid',
                  borderColor: (theme) => alpha('#ed6c02', theme.palette.mode === 'dark' ? 0.35 : 0.25),
                  bgcolor: (theme) => alpha('#ed6c02', theme.palette.mode === 'dark' ? 0.12 : 0.05),
                }}
              >
                <Typography variant="body2" sx={{ fontWeight: 600 }}>
                  {it.caseId}
                </Typography>
                <Typography variant="caption" color="text.secondary" display="block">
                  {it.modelType} · score {Number.isFinite(it.riskScore) ? it.riskScore.toFixed(3) : '—'}
                </Typography>
              </Box>
            ))}
          </DashboardPanel>
        </Grid>

        <Grid item xs={12} md={6}>
          <DashboardPanel
            accent="#00897b"
            icon={<ModelTrainingOutlinedIcon fontSize="small" />}
            title="Recent Training Tasks"
            subheader="点击左侧行查看详情；运行中或排队任务右侧有「停止」按钮（自动刷新）"
            action={
              <FormControl size="small" sx={{ minWidth: 180 }}>
                <InputLabel id="range-label">时间范围</InputLabel>
                <Select
                  labelId="range-label"
                  label="时间范围"
                  value={timeRangeMin}
                  onChange={(e) => setTimeRangeMin(Number(e.target.value))}
                >
                  <MenuItem value={15}>最近 15 分钟</MenuItem>
                  <MenuItem value={60}>最近 1 小时</MenuItem>
                  <MenuItem value={360}>最近 6 小时</MenuItem>
                  <MenuItem value={1440}>最近 24 小时</MenuItem>
                  <MenuItem value={0}>全部</MenuItem>
                </Select>
              </FormControl>
            }
          >
            {recentTasks.length === 0 ? (
              <Typography variant="body2" color="text.secondary">
                当前时间范围内暂无任务
              </Typography>
            ) : (
              <Box
                sx={{
                  borderRadius: 1,
                  border: '1px solid',
                  borderColor: 'divider',
                  bgcolor: 'background.paper',
                  overflow: 'visible',
                }}
              >
                {recentTasks.map((t) => (
                  <Box
                    key={t.taskId}
                    sx={{
                      display: 'flex',
                      alignItems: 'stretch',
                      borderBottom: '1px solid',
                      borderColor: 'divider',
                      '&:last-of-type': { borderBottom: 'none' },
                    }}
                  >
                    <ListItemButton
                      dense
                      selected={selectedTaskId === t.taskId}
                      onClick={() => {
                        setSelectedTaskId(t.taskId)
                        setSelectedTask(null)
                        setSelectedLog('')
                        setDetailOpen(true)
                      }}
                      sx={{
                        flex: 1,
                        minWidth: 0,
                        py: 1,
                        '&.Mui-selected': {
                          bgcolor: (theme) => alpha('#00897b', theme.palette.mode === 'dark' ? 0.28 : 0.12),
                          borderLeft: '3px solid #00897b',
                        },
                      }}
                    >
                      <ListItemText
                        primary={`${t.name || `${t.cancer || ''} ${t.modelType || ''} Training`}`.trim()}
                        secondary={
                          <span>
                            <b>{t.status}</b>  ·  Progress: {fmtPct(t.progress)}  ·  Loss: {t.loss ?? '—'}  ·  C-Index:{' '}
                            {t.cIndex ?? '—'}
                          </span>
                        }
                      />
                    </ListItemButton>
                    {trainingTaskCanStop(t) ? (
                      <Box
                        sx={{
                          display: 'flex',
                          alignItems: 'center',
                          flexShrink: 0,
                          px: 0.75,
                          borderLeft: '1px solid',
                          borderColor: 'divider',
                          bgcolor: (theme) => alpha(theme.palette.warning.main, theme.palette.mode === 'dark' ? 0.12 : 0.06),
                        }}
                      >
                        <Button
                          size="small"
                          variant="contained"
                          color="warning"
                          disabled={Boolean(stoppingTaskId)}
                          startIcon={
                            stoppingTaskId === t.taskId ? (
                              <CircularProgress size={14} color="inherit" />
                            ) : (
                              <StopOutlinedIcon sx={{ fontSize: 18 }} />
                            )
                          }
                          onClick={(e) => {
                            e.preventDefault()
                            e.stopPropagation()
                            stopTrainingTask(t.taskId)
                          }}
                          sx={{ whiteSpace: 'nowrap', minWidth: 72 }}
                        >
                          {stoppingTaskId === t.taskId ? '停止中' : '停止'}
                        </Button>
                      </Box>
                    ) : null}
                  </Box>
                ))}
              </Box>
            )}
          </DashboardPanel>
        </Grid>

        <Grid item xs={12} md={6}>
          <DashboardPanel
            accent="#7b1fa2"
            icon={<AssessmentOutlinedIcon fontSize="small" />}
            title="Evaluation Runs (recent)"
            subheader="点击 run 查看摘要与曲线统计详情"
          >
            {completedRuns.length === 0 ? (
              <Typography variant="body2" color="text.secondary">
                暂无 completed runs
              </Typography>
            ) : (
              <List
                dense
                disablePadding
                sx={{
                  borderRadius: 1,
                  overflow: 'hidden',
                  border: '1px solid',
                  borderColor: 'divider',
                  bgcolor: 'background.paper',
                }}
              >
                {completedRuns.map((r) => (
                  <ListItemButton
                    key={r.taskId}
                    selected={selectedRunId === r.taskId}
                    onClick={async () => {
                      setSelectedRunId(r.taskId)
                      setSelectedRunCurves(null)
                      setRunDetailOpen(true)
                      await loadRunDetail(r.taskId)
                    }}
                    sx={{
                      '&.Mui-selected': {
                        bgcolor: (theme) => alpha('#7b1fa2', theme.palette.mode === 'dark' ? 0.28 : 0.12),
                        borderLeft: '3px solid #7b1fa2',
                      },
                    }}
                  >
                    <ListItemText
                      primary={`${r.taskId} — ${r.modelType}`}
                      secondary={`status: ${r.status} · ROC AUC: ${r.rocAuc ?? r.cIndex ?? '—'}`}
                    />
                  </ListItemButton>
                ))}
              </List>
            )}
          </DashboardPanel>
        </Grid>

        <Dialog open={detailOpen} onClose={() => setDetailOpen(false)} maxWidth="lg" fullWidth>
          <DialogTitle sx={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between', gap: 2 }}>
            <span>训练任务详情</span>
            <Box sx={{ display: 'flex', gap: 1, flexWrap: 'wrap' }}>
              <Chip size="small" label={`状态: ${selectedTask?.status || '—'}`} color={selectedTask?.status === 'completed' ? 'success' : 'default'} />
              <Chip size="small" label={`数据集: ${selectedTask?.cancer || '—'}`} />
              <Chip size="small" label={`模型: ${selectedTask?.modelType || '—'}`} />
              <Chip size="small" label={`退出码: ${selectedTask?.exitCode ?? '—'}`} />
            </Box>
          </DialogTitle>
          <DialogContent dividers>
            {!selectedTask ? (
              <Typography variant="body2" color="text.secondary">
                正在加载…
              </Typography>
            ) : (
              <>
                <Typography variant="body2" sx={{ mb: 1 }}>
                  Task ID
                </Typography>
                <Typography variant="body2" sx={{ mb: 2 }}>
                  <code>{selectedTask.taskId}</code>
                </Typography>

                <Typography variant="subtitle2" sx={{ mb: 1 }}>
                  指标
                </Typography>
                <Grid container spacing={2} sx={{ mb: 2 }}>
                  <Grid item xs={12} md={4}>
                    <Typography variant="body2" color="text.secondary">
                      Epoch
                    </Typography>
                    <Typography variant="body1">
                      {selectedTask.epoch ?? '—'} / {selectedTask.maxEpochs ?? '—'}
                    </Typography>
                  </Grid>
                  <Grid item xs={12} md={4}>
                    <Typography variant="body2" color="text.secondary">
                      Progress
                    </Typography>
                    <Typography variant="body1">{fmtPct(selectedTask.progress)}</Typography>
                  </Grid>
                  <Grid item xs={12} md={4}>
                    <Typography variant="body2" color="text.secondary">
                      Train Loss
                    </Typography>
                    <Typography variant="body1">{selectedTask.loss ?? '—'}</Typography>
                  </Grid>
                  <Grid item xs={12} md={4}>
                    <Typography variant="body2" color="text.secondary">
                      C-Index
                    </Typography>
                    <Typography variant="body1">{selectedTask.cIndex ?? '—'}</Typography>
                  </Grid>
                  <Grid item xs={12} md={4}>
                    <Typography variant="body2" color="text.secondary">
                      Learning rate
                    </Typography>
                    <Typography variant="body1">{selectedTask.learningRate ?? '—'}</Typography>
                  </Grid>
                  <Grid item xs={12} md={4}>
                    <Typography variant="body2" color="text.secondary">
                      Batch size
                    </Typography>
                    <Typography variant="body1">
                      {Number(selectedTask.batchSize) > 0 ? selectedTask.batchSize : '—（MIL 未用 batch）'}
                    </Typography>
                  </Grid>
                </Grid>

                <Typography variant="subtitle2" sx={{ mb: 1 }}>
                  时间
                </Typography>
                <Typography variant="body2" color="text.secondary" sx={{ mb: 2 }}>
                  开始: {selectedTask.startedAt ?? '—'} {' | '} 结束: {selectedTask.endedAt ?? '—'}
                </Typography>

                {(() => {
                  const cfg = buildTaskConfigSnapshot(selectedTask)
                  if (!cfg) return null
                  const fusionLabel =
                    cfg.ensembleFusion === 'gate'
                      ? '门控加权 (gate)'
                      : cfg.ensembleFusion === 'concat'
                        ? '对齐后拼接 (concat)'
                        : cfg.ensembleFusion || '—'
                  const kv = (label, val) => (
                    <Grid item xs={12} sm={6} md={4} key={label}>
                      <Typography variant="caption" color="text.secondary" display="block">
                        {label}
                      </Typography>
                      <Typography variant="body2" sx={{ fontWeight: 600 }}>
                        {val}
                      </Typography>
                    </Grid>
                  )
                  return (
                    <>
                      <Typography variant="subtitle2" sx={{ mb: 1 }}>
                        训练参数与消融（本次任务）
                      </Typography>
                      <Box
                        sx={{
                          mb: 2,
                          p: 1.5,
                          borderRadius: 1,
                          border: '1px solid',
                          borderColor: 'divider',
                          bgcolor: (theme) => alpha(theme.palette.primary.main, theme.palette.mode === 'dark' ? 0.12 : 0.06),
                        }}
                      >
                        <Typography variant="caption" color="text.secondary" sx={{ display: 'block', mb: 1 }}>
                          以下由任务记录与启动命令综合得出；旧任务若缺少部分字段，会尽量从命令行反查。
                        </Typography>
                        <Grid container spacing={1.5}>
                          {kv('癌种 (cancer)', cfg.cancer)}
                          {kv('模式 (mode)', cfg.mode)}
                          {kv('最大轮数 (maxEpochs)', cfg.maxEpochs)}
                          {kv('学习率 (lr)', cfg.learningRate)}
                          {kv('交叉验证折数 (k)', cfg.kFolds)}
                          {kv('权重衰减 (reg)', cfg.weightDecay)}
                          {kv('随机种子 (seed)', cfg.seed)}
                          {kv('早停 (--early_stopping)', cfg.earlyStopping ? '开启' : '关闭')}
                          {kv(
                            'Batch size',
                            Number(cfg.batchSize) > 0 ? String(cfg.batchSize) : '—（MIL 通常为 0）'
                          )}
                          {cfg.repeatTotal != null && Number(cfg.repeatTotal) > 1
                            ? kv('重复训练轮次 (repeat)', String(cfg.repeatTotal))
                            : null}
                          {String(cfg.model) === 'EnsembleFeature' ? (
                            <>
                              <Grid item xs={12}>
                                <Divider sx={{ my: 0.5 }} />
                              </Grid>
                              <Grid item xs={12}>
                                <Typography variant="caption" color="primary" sx={{ fontWeight: 700 }}>
                                  EnsembleFeature 消融选项
                                </Typography>
                              </Grid>
                              {kv('融合方式', fusionLabel)}
                              {kv(
                                '冻结五基线（仅训对齐/门控/融合）',
                                cfg.finetuneEnsemble ? '否（已开启端到端微调 finetune_ensemble）' : '是'
                              )}
                              {kv(
                                '排除的基线特征 (ensembleExclude)',
                                cfg.ensembleExclude.length > 0 ? cfg.ensembleExclude.join('、') : '无（五路均参与）'
                              )}
                              {kv(
                                '手动基线权重目录',
                                cfg.ensembleCkptDir ? <code style={{ wordBreak: 'break-all' }}>{cfg.ensembleCkptDir}</code> : '—（走 best_models / tasks 自动解析）'
                              )}
                            </>
                          ) : null}
                        </Grid>
                      </Box>
                    </>
                  )
                })()}

                <Typography variant="subtitle2" sx={{ mb: 1 }}>
                  启动命令
                </Typography>
                <Box
                  sx={{
                    border: '1px solid rgba(0,0,0,0.12)',
                    borderRadius: 1,
                    p: 1.5,
                    mb: 2,
                    overflowX: 'auto',
                    fontFamily: 'ui-monospace, SFMono-Regular, Menlo, Monaco, Consolas, monospace',
                    fontSize: 12,
                    background: '#fafafa',
                  }}
                >
                  {selectedTask.command || '—'}
                </Box>

                <Box sx={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between', mb: 1 }}>
                  <Typography variant="subtitle2">训练日志（末尾最多 2000 行）</Typography>
                  <Button variant="text" onClick={() => selectedTaskId && loadTaskDetail(selectedTaskId)}>
                    刷新日志
                  </Button>
                </Box>
                <Box
                  sx={{
                    whiteSpace: 'pre-wrap',
                    fontFamily: 'ui-monospace, SFMono-Regular, Menlo, Monaco, Consolas, monospace',
                    fontSize: 12,
                    background: '#0b1020',
                    color: '#d7e3ff',
                    borderRadius: 1,
                    p: 2,
                    maxHeight: 520,
                    overflow: 'auto',
                  }}
                >
                  {selectedLog || '—'}
                </Box>
              </>
            )}
          </DialogContent>
          <DialogActions sx={{ justifyContent: 'space-between', flexWrap: 'wrap', gap: 1 }}>
            {selectedTask && trainingTaskCanStop(selectedTask) ? (
              <Button
                color="warning"
                variant="outlined"
                startIcon={<StopOutlinedIcon />}
                disabled={Boolean(stoppingTaskId)}
                onClick={() => stopTrainingTask(selectedTask.taskId)}
              >
                {stoppingTaskId === selectedTask.taskId ? '正在停止…' : '停止该任务'}
              </Button>
            ) : (
              <span />
            )}
            <Button onClick={() => setDetailOpen(false)}>关闭</Button>
          </DialogActions>
        </Dialog>

        <Dialog open={runDetailOpen} onClose={() => setRunDetailOpen(false)} maxWidth="md" fullWidth>
          <DialogTitle sx={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between', gap: 2 }}>
            <span>评估 Run 详情</span>
            <Box sx={{ display: 'flex', gap: 1, flexWrap: 'wrap' }}>
              <Chip size="small" label={`taskId: ${selectedRunId || '—'}`} />
              <Chip size="small" label={`模型: ${selectedRunMeta?.modelType || '—'}`} />
              <Chip size="small" label={`状态: ${selectedRunMeta?.status || '—'}`} />
            </Box>
          </DialogTitle>
          <DialogContent dividers>
            {runDetailLoading ? (
              <Typography variant="body2" color="text.secondary">
                正在加载评估详情…
              </Typography>
            ) : !selectedRunCurves ? (
              <Typography variant="body2" color="text.secondary">
                暂无评估详情
              </Typography>
            ) : (
              <>
                <Typography variant="subtitle2" sx={{ mb: 1 }}>
                  摘要指标
                </Typography>
                <Grid container spacing={2} sx={{ mb: 2 }}>
                  <Grid item xs={12} md={6}>
                    <Typography variant="body2" color="text.secondary">
                      验证集最佳 ROC AUC
                    </Typography>
                    <Typography variant="body1">
                      {selectedRunCurves?.summary?.bestValRocAuc ??
                        selectedRunCurves?.summary?.bestValCIndex ??
                        '—'}
                    </Typography>
                  </Grid>
                  <Grid item xs={12} md={6}>
                    <Typography variant="body2" color="text.secondary">
                      测试集最终 ROC AUC
                    </Typography>
                    <Typography variant="body1">
                      {selectedRunCurves?.summary?.finalTestRocAuc ??
                        selectedRunCurves?.summary?.finalTestCIndex ??
                        '—'}
                    </Typography>
                  </Grid>
                  <Grid item xs={12} md={6}>
                    <Typography variant="body2" color="text.secondary">
                      训练集最终 ROC AUC
                    </Typography>
                    <Typography variant="body1">
                      {selectedRunCurves?.summary?.finalTrainRocAuc ??
                        selectedRunCurves?.summary?.finalTrainCIndex ??
                        '—'}
                    </Typography>
                  </Grid>
                  <Grid item xs={12} md={6}>
                    <Typography variant="body2" color="text.secondary">
                      记录点数
                    </Typography>
                    <Typography variant="body1">{selectedRunCurves?.summary?.epochCount ?? '—'}</Typography>
                  </Grid>
                </Grid>
                <Typography variant="subtitle2" sx={{ mb: 1 }}>
                  最近曲线点（最多 5 条）
                </Typography>
                {(selectedRunCurves?.series || []).slice(-5).map((p) => (
                  <Typography key={p.epoch} variant="body2" color="text.secondary" sx={{ mb: 0.5 }}>
                    epoch {p.epoch} · trainLoss {p.trainLoss ?? '—'} · valLoss {p.valLoss ?? '—'} · trainRocAuc{' '}
                    {p.trainRocAuc ?? p.trainCIndex ?? '—'} · valRocAuc {p.valRocAuc ?? p.valCIndex ?? '—'}
                  </Typography>
                ))}
              </>
            )}
          </DialogContent>
          <DialogActions>
            <Button onClick={() => setRunDetailOpen(false)}>关闭</Button>
          </DialogActions>
        </Dialog>
      </Grid>
    </Box>
  )
}

