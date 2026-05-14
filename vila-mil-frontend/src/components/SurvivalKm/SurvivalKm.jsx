import React, { useEffect, useMemo, useState } from 'react'
import { Link as RouterLink } from 'react-router-dom'
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
  MenuItem,
  Select,
  Stack,
  Typography,
} from '@mui/material'
import { alpha } from '@mui/material/styles'
import { CartesianGrid, Legend, Line, LineChart, ResponsiveContainer, Tooltip, XAxis, YAxis } from 'recharts'

import { useCohortAndKmFromPredictions } from '../../hooks/useCohortAndKmFromPredictions'
import { trainingApi } from '../../services/api'
import { formatPredictionTaskMenuLabel, enrichEnsembleTrainingTitle } from '../../utils/ensembleTaskLabel'
import {
  KM_SIX_MODEL_ORDER,
  fmtFixed,
  fmtLogRankP,
  kmCurveKey,
  kmStrokeForModel,
} from '../../utils/kmChartUtils'
import {
  cohortHazardRatioCellText,
  kmChartPanelSx,
  kmSectionCardSx,
  kmSixSectionCardSx,
  shortTaskId,
} from '../../utils/predictionCohortKmUtils'

export default function SurvivalKm() {
  const [tasks, setTasks] = useState([])
  const [taskId, setTaskId] = useState('')

  const loadTasks = async () => {
    try {
      const tRes = await trainingApi.history()
      setTasks(tRes?.tasks || tRes?.data?.tasks || [])
    } catch {
      setTasks([])
    }
  }

  useEffect(() => {
    loadTasks()
  }, [])

  const availableTasks = useMemo(
    () => (tasks || []).filter((t) => t.status === 'completed' && Boolean(t.hasCheckpoint)),
    [tasks]
  )

  const selectedTask = useMemo(
    () => availableTasks.find((t) => String(t.taskId || t.id || '').trim() === String(taskId || '').trim()),
    [availableTasks, taskId]
  )

  const effectiveTaskId = String(taskId || '').trim()
  const effectiveModelType = String(selectedTask?.modelType || selectedTask?.model_type || '').trim()
  const cancer = String(selectedTask?.cancer || selectedTask?.cancerType || 'LUSC').trim() || 'LUSC'

  const ck = useCohortAndKmFromPredictions({
    effectiveTaskId,
    effectiveModelType,
    cancer,
    availableTasks,
    tasks,
  })

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
              ? 'linear-gradient(115deg, rgba(46,125,50,0.22) 0%, rgba(21,101,192,0.12) 100%)'
              : 'linear-gradient(115deg, rgba(46,125,50,0.10) 0%, rgba(21,101,192,0.06) 100%)',
        })}
      >
        <Stack direction={{ xs: 'column', sm: 'row' }} spacing={1.5} alignItems={{ sm: 'center' }} justifyContent="space-between">
          <Typography variant="h4" sx={{ fontWeight: 700 }}>
            生存曲线（KM）
          </Typography>
          <Button component={RouterLink} to="/prediction" size="small" variant="outlined" color="inherit">
            前往 Prediction
          </Button>
        </Stack>
        <Typography variant="body2" color="text.secondary" sx={{ mt: 1 }}>
          六模型最佳任务同图总体 KM，以及按所选训练任务的「预测 + 随访」分层 KM。任务列表与 Prediction 使用同一套已完成 checkpoint。
        </Typography>
      </Box>

      <Card sx={{ mb: 3, borderRadius: 2, border: '1px solid', borderColor: 'divider' }}>
        <CardHeader title="选择训练任务（用于下方「预测 + 随访」KM）" />
        <CardContent>
          <Stack direction={{ xs: 'column', sm: 'row' }} spacing={2} alignItems={{ sm: 'center' }}>
            <FormControl sx={{ minWidth: 360, maxWidth: '100%' }}>
              <InputLabel id="sk-task-label">Task</InputLabel>
              <Select
                labelId="sk-task-label"
                label="Task"
                value={taskId && availableTasks.some((t) => String(t.taskId || t.id) === String(taskId)) ? taskId : ''}
                onChange={(e) => setTaskId(e.target.value)}
              >
                {availableTasks.map((t) => {
                  const id = String(t.taskId || t.id || '')
                  return (
                    <MenuItem key={id} value={id}>
                      {formatPredictionTaskMenuLabel(t)}
                    </MenuItem>
                  )
                })}
              </Select>
            </FormControl>
            <Button variant="outlined" onClick={loadTasks}>
              刷新任务
            </Button>
          </Stack>
          {!availableTasks.length ? (
            <Typography variant="body2" color="text.secondary" sx={{ mt: 1.5 }}>
              暂无已完成且含 checkpoint 的任务，请先在 Training 完成训练。
            </Typography>
          ) : null}
        </CardContent>
      </Card>

      {ck.cohortCIndexByTask && ck.cohortCIndexByTask.length > 0 ? (
        <Card sx={(theme) => kmSixSectionCardSx(theme)}>
          <CardHeader title="六模型最佳任务 · Kaplan–Meier 总体曲线（同图）" />
          <CardContent>
            {ck.kmSixError ? (
              <Alert severity="warning" sx={{ mb: 1.5 }}>
                {ck.kmSixError}
              </Alert>
            ) : null}
            <Box
              sx={{
                display: 'grid',
                gridTemplateColumns: { xs: '1fr', sm: 'repeat(2, 1fr)', md: 'repeat(3, 1fr)' },
                gap: 1.25,
                mb: 2,
              }}
            >
              {KM_SIX_MODEL_ORDER.map((mt) => {
                const c = (ck.kmSixCurvesDisplay || []).find((x) => String(x.modelType) === mt)
                const stroke = kmStrokeForModel(mt)
                return (
                  <Box
                    key={mt}
                    sx={{
                      p: 1.25,
                      borderRadius: 1,
                      border: '1px solid',
                      borderColor: 'divider',
                      borderLeft: `4px solid ${stroke}`,
                      bgcolor: (theme) => alpha(stroke, theme.palette.mode === 'dark' ? 0.12 : 0.06),
                    }}
                  >
                    <Stack direction="row" alignItems="center" spacing={1} flexWrap="wrap" useFlexGap>
                      <Typography variant="subtitle2" sx={{ fontWeight: 800, color: stroke }}>
                        {mt}
                      </Typography>
                      <Typography variant="caption" color="text.secondary" sx={{ fontFamily: 'ui-monospace, monospace' }}>
                        {c?.taskId ? shortTaskId(c.taskId) : '—'}
                      </Typography>
                    </Stack>
                    <Typography variant="caption" color="text.secondary" display="block" sx={{ mt: 0.5 }}>
                      n={c?.n != null ? c.n : '—'}
                      {c?.cIndex != null && Number.isFinite(Number(c.cIndex)) ? ` · C-index ${Number(c.cIndex).toFixed(4)}` : ''}
                    </Typography>
                    <Typography variant="body2" sx={{ mt: 0.75, fontFamily: 'ui-monospace, monospace', fontSize: 13 }}>
                      HR {cohortHazardRatioCellText(c || {})}
                    </Typography>
                    {c?.logRankP != null && Number.isFinite(Number(c.logRankP)) ? (
                      <Typography variant="caption" color="text.secondary" display="block">
                        Log-rank p={fmtLogRankP(c.logRankP)}
                      </Typography>
                    ) : null}
                    {c?.messageZh ? (
                      <Typography variant="caption" color="warning.main" display="block" sx={{ mt: 0.5 }}>
                        {c.messageZh}
                      </Typography>
                    ) : null}
                  </Box>
                )
              })}
            </Box>
            {ck.kmSixChartData.length > 0 && ck.kmSixCurvesWithData.length > 0 ? (
              <Box sx={(theme) => kmChartPanelSx(theme)}>
                <ResponsiveContainer width="100%" height="100%">
                  <LineChart data={ck.kmSixChartData} margin={{ top: 8, right: 12, left: 4, bottom: 4 }}>
                    <CartesianGrid strokeDasharray="3 3" />
                    <XAxis dataKey="time" type="number" tickFormatter={(v) => fmtFixed(v, 2)} label={{ value: '时间', position: 'insideBottom', offset: -2 }} />
                    <YAxis domain={[0, 1.05]} tickFormatter={(v) => fmtFixed(v, 2)} width={48} label={{ value: 'S(t)', angle: -90, position: 'insideLeft' }} />
                    <Tooltip
                      formatter={(value, name) => [fmtFixed(value, 4), name]}
                      labelFormatter={(t) => `time ${fmtFixed(t, 3)}`}
                    />
                    <Legend />
                    {ck.kmSixCurvesWithData.map((c) => {
                      const mt = String(c.modelType || c.label || '')
                      const dk = kmCurveKey(c.label || mt)
                      return (
                        <Line
                          key={mt}
                          type="stepAfter"
                          dataKey={dk}
                          name={mt}
                          stroke={kmStrokeForModel(mt)}
                          strokeWidth={2}
                          dot={false}
                          isAnimationActive={false}
                        />
                      )
                    })}
                  </LineChart>
                </ResponsiveContainer>
              </Box>
            ) : !ck.kmSixLoading ? (
              <Typography variant="body2" color="text.secondary">
                暂无足够数据绘制六模型曲线（需各模型有可计算 C-index 的代表任务且随访配对 n≥2）。
              </Typography>
            ) : null}
            {ck.kmSixLoading ? (
              <Box sx={{ display: 'flex', alignItems: 'center', gap: 1, mt: 1 }}>
                <CircularProgress size={22} />
                <Typography variant="body2" color="text.secondary">
                  正在加载六模型 KM…
                </Typography>
              </Box>
            ) : null}
          </CardContent>
        </Card>
      ) : null}

      {effectiveTaskId ? (
        <Card sx={(theme) => kmSectionCardSx(theme)}>
          <CardHeader title="Kaplan–Meier 生存曲线（本系统预测 + 随访）" />
          <CardContent>
            <Stack direction="row" spacing={1} flexWrap="wrap" useFlexGap sx={{ mb: 1.5 }} alignItems="center">
              <Chip size="small" variant="outlined" label={`taskId ${shortTaskId(effectiveTaskId)}`} />
              {(() => {
                const kChip = ck.kmFromTaskForChips || ck.kmFromTask
                return (
                  <>
                    {ck.kmFromTask?.stratificationKind ? (
                      <Chip
                        size="small"
                        color="secondary"
                        variant="outlined"
                        label={
                          ck.kmFromTask.stratificationKind === 'risk_median'
                            ? '分层：risk 中位数'
                            : ck.kmFromTask.stratificationKind === 'pred_class_quartile'
                              ? '分层：predClass 0–1 / 2–3'
                              : ck.kmFromTask.stratificationKind === 'rank_half'
                                ? '分层：排序均分'
                                : `分层：${ck.kmFromTask.stratificationKind}`
                        }
                      />
                    ) : null}
                    {kChip?.logRankP != null && Number.isFinite(Number(kChip.logRankP)) ? (
                      <Chip size="small" variant="outlined" label={`Log-rank p=${fmtLogRankP(kChip.logRankP)}`} />
                    ) : null}
                    {ck.kmFromTask?.counts ? (
                      <Chip
                        size="small"
                        variant="outlined"
                        label={`n=${ck.kmFromTask.counts.nTotal ?? '—'}（低/高 ${ck.kmFromTask.counts.nLow ?? '—'}/${ck.kmFromTask.counts.nHigh ?? '—'}）`}
                      />
                    ) : null}
                    {kChip?.hazardRatio != null && Number.isFinite(Number(kChip.hazardRatio)) ? (
                      <Chip
                        size="small"
                        variant="outlined"
                        color="primary"
                        label={`HR ${fmtFixed(kChip.hazardRatio, 3)}${
                          kChip.hazardRatio95Ci &&
                          Array.isArray(kChip.hazardRatio95Ci) &&
                          kChip.hazardRatio95Ci.length >= 2
                            ? `（${fmtFixed(kChip.hazardRatio95Ci[0], 3)}–${fmtFixed(kChip.hazardRatio95Ci[1], 3)}）`
                            : ''
                        }`}
                      />
                    ) : null}
                    {kChip?.hazardRatioP != null && Number.isFinite(Number(kChip.hazardRatioP)) ? (
                      <Chip size="small" variant="outlined" label={`Cox p=${fmtLogRankP(kChip.hazardRatioP)}`} />
                    ) : null}
                    {kChip?.cohortCIndex != null && Number.isFinite(Number(kChip.cohortCIndex)) ? (
                      <Chip
                        size="small"
                        variant="outlined"
                        label={`C-index ${fmtFixed(kChip.cohortCIndex, 4)}（可比对 ${kChip.comparablePairs ?? '—'}）`}
                      />
                    ) : null}
                  </>
                )
              })()}
              {ck.kmLoading ? <CircularProgress size={22} sx={{ ml: 0.5 }} /> : null}
            </Stack>
            {ck.cohortSummaryForSelectedTask ? (
              <Typography variant="caption" color="text.secondary" sx={{ display: 'block', mb: 1 }}>
                训练任务展示名：{ck.resolveCohortTaskLabel(ck.cohortSummaryForSelectedTask)} · 模型{' '}
                <strong>{ck.cohortSummaryForSelectedTask.modelType ?? '—'}</strong>
              </Typography>
            ) : (
              (() => {
                const tid = String(effectiveTaskId || '').trim()
                const tm = ck.latestTaskMetaById.get(tid)
                const lab =
                  String(tm?.modelType || tm?.model_type || effectiveModelType || '') === 'EnsembleDecision'
                    ? enrichEnsembleTrainingTitle({
                        modelType: 'EnsembleDecision',
                        taskLabel: tm?.name,
                        name: tm?.name,
                        cancer: String(tm?.cancer || tm?.cancerType || '').trim() || cancer,
                        ensembleExclude: tm?.ensembleExclude,
                      })
                    : ''
                return (
                  <Typography variant="caption" color="text.secondary" sx={{ display: 'block', mb: 1 }}>
                    训练任务：{lab || tm?.name || tid}
                    {tm?.modelType || tm?.model_type ? (
                      <>
                        {' '}
                        · 模型 <strong>{String(tm.modelType || tm.model_type)}</strong>
                      </>
                    ) : null}
                  </Typography>
                )
              })()
            )}
            {ck.kmFromTask?.splitDescriptionZh ? (
              <Alert severity="info" sx={{ mb: 1.5 }}>
                {ck.kmFromTask.splitDescriptionZh}
              </Alert>
            ) : null}
            {ck.kmFromTask?.hazardRatioRefZh ? (
              <Typography variant="caption" color="text.secondary" sx={{ display: 'block', mb: 1 }}>
                HR 参照：{ck.kmFromTask.hazardRatioRefZh}
              </Typography>
            ) : null}
            {ck.kmError ? (
              <Alert severity={ck.kmFromTask?.ok === false ? 'warning' : 'error'} sx={{ mb: 1.5 }}>
                {ck.kmError}
              </Alert>
            ) : null}
            {ck.kmFromTask?.noteZh ? (
              <Typography variant="caption" color="text.secondary" sx={{ display: 'block', mb: 1 }}>
                {ck.kmFromTask.noteZh}
              </Typography>
            ) : null}
            {ck.kmChartData.length > 0 && ck.kmCurveLines.length > 0 ? (
              <Box sx={(theme) => kmChartPanelSx(theme)}>
                <ResponsiveContainer width="100%" height="100%">
                  <LineChart data={ck.kmChartData} margin={{ top: 8, right: 12, left: 4, bottom: 4 }}>
                    <CartesianGrid strokeDasharray="3 3" />
                    <XAxis dataKey="time" type="number" tickFormatter={(v) => fmtFixed(v, 2)} label={{ value: '时间', position: 'insideBottom', offset: -2 }} />
                    <YAxis domain={[0, 1.05]} tickFormatter={(v) => fmtFixed(v, 2)} width={48} label={{ value: 'S(t)', angle: -90, position: 'insideLeft' }} />
                    <Tooltip
                      formatter={(value, name) => [fmtFixed(value, 4), name]}
                      labelFormatter={(t) => `time ${fmtFixed(t, 3)}`}
                    />
                    <Legend />
                    {ck.kmCurveLines.map((line) => (
                      <Line
                        key={line.dataKey}
                        type="stepAfter"
                        dataKey={line.dataKey}
                        name={line.label}
                        stroke={line.stroke}
                        strokeWidth={2}
                        dot={false}
                        isAnimationActive={false}
                      />
                    ))}
                  </LineChart>
                </ResponsiveContainer>
              </Box>
            ) : !ck.kmLoading && effectiveTaskId ? (
              <Typography variant="body2" color="text.secondary">
                暂无可用曲线数据（多为随访配对不足或尚未对该任务写入预测）。完成预测并维护 Clinical 的 <code>time</code>、<code>status</code> 后将自动显示。
              </Typography>
            ) : null}
          </CardContent>
        </Card>
      ) : (
        <Alert severity="info" sx={{ mb: 2 }}>
          请在上方选择一个训练任务以查看「预测 + 随访」Kaplan–Meier 曲线。
        </Alert>
      )}
    </Box>
  )
}
