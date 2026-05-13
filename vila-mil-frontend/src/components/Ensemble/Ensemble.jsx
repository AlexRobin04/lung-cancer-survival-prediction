import React, { useEffect, useState, useCallback, useMemo } from 'react'
import {
  Alert,
  Box,
  Button,
  Card,
  CardContent,
  CardHeader,
  Chip,
  CircularProgress,
  Divider,
  FormControl,
  FormControlLabel,
  InputLabel,
  MenuItem,
  Select,
  Stack,
  Checkbox,
  Switch,
  TextField,
  Typography,
  Link as MuiLink,
} from '@mui/material'
import { alpha } from '@mui/material/styles'
import { Link } from 'react-router-dom'
import { trainingApi } from '../../services/api'
import useCancerOptions from '../../hooks/useCancerOptions'
import { CANCER_CN_MAP } from '../../constants/trainingOptions'
import { ENSEMBLE_BRANCH_IDS, BRANCH_SUBSET_PRESETS } from '../../constants/ensembleBranchConfig'
import {
  readEnsembleTiebreakAllowFallback,
  writeEnsembleTiebreakAllowFallback,
} from '../../constants/ensemblePredictPrefs'
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

/**
 * 决策级集成（EnsembleDecision）：与基线 MIL「端到端多 epoch 训练」分离。
 * 仍调用 POST /api/training/start，但配置语义为加载五路 checkpoint、搜索融合权重并写集成 ckpt。
 */
export default function Ensemble() {
  const [error, setError] = useState('')
  const [notice, setNotice] = useState('')
  const [loading, setLoading] = useState(false)

  const { cancerOptions, cancer, setCancer } = useCancerOptions('LUSC')
  const [maxEpochs, setMaxEpochs] = useState(120)
  const [learningRate, setLearningRate] = useState(1e-5)
  const [seed, setSeed] = useState(1)
  const [kFolds, setKFolds] = useState(4)
  const [weightDecay, setWeightDecay] = useState(1e-5)
  const [earlyStopping, setEarlyStopping] = useState(false)

  const [decisionFusion] = useState('avg_prob')
  const [ensembleBranchPriorAuto, setEnsembleBranchPriorAuto] = useState(true)
  const [ensembleBranchPriorTemperature, setEnsembleBranchPriorTemperature] = useState('0.55')
  /** 蒸馏+tie-break 路径：验证子集 C-index 不足阈值时是否将 λ 回退为 0（与 Prediction 请求联动） */
  const [tiebreakFallbackEnabled, setTiebreakFallbackEnabled] = useState(() => readEnsembleTiebreakAllowFallback())
  const [decisionBranchWeightById, setDecisionBranchWeightById] = useState(() =>
    Object.fromEntries(ENSEMBLE_BRANCH_IDS.map((b) => [b, '']))
  )
  const [branchInclude, setBranchInclude] = useState(() =>
    Object.fromEntries(ENSEMBLE_BRANCH_IDS.map((b) => [b, true]))
  )

  useEffect(() => {
    writeEnsembleTiebreakAllowFallback(tiebreakFallbackEnabled)
  }, [tiebreakFallbackEnabled])

  useEffect(() => {
    setDecisionBranchWeightById((prev) => {
      const next = { ...prev }
      let changed = false
      for (const b of ENSEMBLE_BRANCH_IDS) {
        if (!branchInclude[b] && String(next[b] ?? '').trim() !== '') {
          next[b] = ''
          changed = true
        }
      }
      return changed ? next : prev
    })
  }, [branchInclude])

  const formatCancerLabel = (code) => {
    const k = String(code || '').trim()
    if (!k) return ''
    const cn = CANCER_CN_MAP?.[k]
    return cn ? `${k}（${cn}）` : k
  }

  const attachEnsemblePriorApiFields = useCallback(
    (payload) => {
      payload.ensembleBranchPriorAuto = !!ensembleBranchPriorAuto
      const tRaw = String(ensembleBranchPriorTemperature ?? '').trim()
      if (tRaw !== '') {
        const n = Number(tRaw)
        if (!Number.isNaN(n) && n > 0) payload.ensembleBranchPriorTemperature = n
      }
    },
    [ensembleBranchPriorAuto, ensembleBranchPriorTemperature]
  )

  const startEnsemble = async () => {
    const ex = ENSEMBLE_BRANCH_IDS.filter((b) => !branchInclude[b])
    if (ex.length >= ENSEMBLE_BRANCH_IDS.length) {
      setError('至少保留一路基线（不可五路全关）')
      return
    }
    setLoading(true)
    setError('')
    setNotice('')
    try {
      const payload = {
        cancer,
        modelType: 'EnsembleDecision',
        mode: 'transformer',
        maxEpochs: Number(maxEpochs),
        learningRate: Number(learningRate),
        kFolds: Math.min(20, Math.max(1, Number(kFolds) || 4)),
        weightDecay: Number(weightDecay) >= 0 ? Number(weightDecay) : 1e-5,
        earlyStopping,
        repeat: 1,
        seed: Number(seed) || 1,
        decisionFusion: String(decisionFusion || 'avg_prob').toLowerCase(),
      }
      if (ex.length > 0) payload.ensembleExclude = ex
      attachEnsemblePriorApiFields(payload)
      if (String(decisionFusion || '').toLowerCase() === 'weighted') {
        const wObj = {}
        for (const b of ENSEMBLE_BRANCH_IDS) {
          if (!branchInclude[b]) continue
          const raw = String(decisionBranchWeightById[b] ?? '').trim()
          if (raw === '') continue
          const n = Number(raw)
          if (Number.isNaN(n)) {
            setError(`${b} 权重须为数字`)
            setLoading(false)
            return
          }
          if (n < 0) {
            setError(`${b} 权重不能为负`)
            setLoading(false)
            return
          }
          wObj[b] = n
        }
        if (Object.keys(wObj).length > 0) {
          payload.decisionBranchWeights = wObj
        }
      }
      const res = await trainingApi.start(payload)
      if (res?.queued) {
        setNotice('当前有任务在运行，集成任务已加入训练队列（与基线共用队列）')
      } else {
        setNotice('集成任务已启动')
      }
    } catch (e) {
      setError(e?.response?.data?.message || e.message || '启动失败')
    } finally {
      setLoading(false)
    }
  }

  const fusionNote = useMemo(
    () =>
      `当前将提交：启用=[${ENSEMBLE_BRANCH_IDS.filter((b) => branchInclude[b]).join(', ') || '（无）'}]；--ensemble_exclude ${ENSEMBLE_BRANCH_IDS.filter((b) => !branchInclude[b]).join(',') || '（空）'}`,
    [branchInclude]
  )

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
              ? 'linear-gradient(115deg, rgba(0,105,92,0.25) 0%, rgba(25,118,210,0.12) 100%)'
              : 'linear-gradient(115deg, rgba(0,137,123,0.14) 0%, rgba(25,118,210,0.06) 100%)',
        })}
      >
        <Typography variant="h4" gutterBottom sx={{ fontWeight: 700, mb: 1 }}>
          Ensemble（决策级集成）
        </Typography>
        <Typography variant="body2" color="text.secondary" sx={{ mb: 1 }}>
          本页仅负责 <strong>EnsembleDecision</strong>：加载五路基线 checkpoint、在验证集上搜索决策权重并保存集成 checkpoint；<strong>不是</strong>与 AMIL/S4 等相同的「多 epoch 端到端训练」。
        </Typography>
        <Typography variant="body2" color="text.secondary">
          请先完成五路基线训练；队列与历史任务仍在{' '}
          <MuiLink component={Link} to="/training" underline="hover">
            Training
          </MuiLink>{' '}
          页查看与停止。
        </Typography>
      </Box>

      {error && (
        <Alert severity="error" sx={{ mb: 2 }} onClose={() => setError('')}>
          {error}
        </Alert>
      )}
      <Toast open={!!notice} message={notice} severity="success" onClose={() => setNotice('')} />

      <Card sx={sectionCardSx('#00695c')}>
        <CardHeader
          title="启动集成任务"
          titleTypographyProps={{ variant: 'subtitle1', fontWeight: 700 }}
        />
        <CardContent>
          <Box
            sx={{
              display: 'grid',
              gridTemplateColumns: { xs: '1fr', md: 'repeat(4, minmax(0, 1fr))' },
              gap: 2,
              alignItems: 'center',
            }}
          >
            <FormControl sx={{ minWidth: 160 }}>
              <InputLabel id="ens-cancer">Cancer</InputLabel>
              <Select labelId="ens-cancer" label="Cancer" value={cancer} onChange={(e) => setCancer(e.target.value)}>
                {cancerOptions.map((opt) => (
                  <MenuItem key={opt.value} value={opt.value}>
                    {formatCancerLabel(opt.label || opt.value)}
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
              helperText="写入任务元数据，供基线对齐；集成子进程不跑满该 epoch 数"
            />
            <TextField
              label="learningRate"
              type="number"
              value={learningRate}
              onChange={(e) => setLearningRate(e.target.value)}
              inputProps={{ step: '0.000001' }}
            />
            <TextField
              label="kFolds"
              type="number"
              value={kFolds}
              onChange={(e) => setKFolds(e.target.value)}
              inputProps={{ min: 1, max: 20, step: 1 }}
            />
            <TextField
              label="weightDecay"
              type="number"
              value={weightDecay}
              onChange={(e) => setWeightDecay(e.target.value)}
              inputProps={{ step: '0.0000001', min: 0 }}
            />
            <TextField label="seed" type="number" value={seed} onChange={(e) => setSeed(e.target.value)} />
            <FormControlLabel
              sx={{ gridColumn: { xs: '1 / -1', md: 'span 2' } }}
              control={<Checkbox checked={earlyStopping} onChange={(e) => setEarlyStopping(e.target.checked)} />}
              label="早停（--early_stopping）"
            />

            <Box sx={{ gridColumn: '1 / -1' }}>
              <Typography variant="subtitle2" sx={{ fontWeight: 600, mb: 0.5 }}>
                基线子集（ensembleExclude）
              </Typography>
              <Typography variant="caption" color="text.secondary" sx={{ display: 'block', mb: 1 }}>
                勾选 = 该路参与决策融合；取消勾选 = 排除该路基线 logits。
              </Typography>
              <Stack direction="row" flexWrap="wrap" gap={1} sx={{ mb: 1 }} alignItems="center">
                {ENSEMBLE_BRANCH_IDS.map((b) => (
                  <FormControlLabel
                    key={b}
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
                  <Chip key={sp.id} size="small" label={sp.label} variant="outlined" onClick={() => setBranchInclude(sp.map())} />
                ))}
                {ENSEMBLE_BRANCH_IDS.map((b) => (
                  <Chip
                    key={`ex-${b}`}
                    size="small"
                    label={`仅排除 ${b}`}
                    variant="outlined"
                    onClick={() =>
                      setBranchInclude(Object.fromEntries(ENSEMBLE_BRANCH_IDS.map((x) => [x, x !== b])))
                    }
                  />
                ))}
                {ENSEMBLE_BRANCH_IDS.map((b) => (
                  <Chip
                    key={`on-${b}`}
                    size="small"
                    color="primary"
                    variant="outlined"
                    label={`仅启用 ${b}`}
                    onClick={() =>
                      setBranchInclude(Object.fromEntries(ENSEMBLE_BRANCH_IDS.map((x) => [x, x === b])))
                    }
                  />
                ))}
              </Stack>
              <Typography variant="caption" color="text.secondary" sx={{ display: 'block' }}>
                {fusionNote}
              </Typography>
            </Box>

            <Box sx={{ gridColumn: '1 / -1' }}>
              <Stack direction={{ xs: 'column', sm: 'row' }} flexWrap="wrap" gap={2} alignItems={{ sm: 'center' }}>
                <FormControlLabel
                  control={
                    <Checkbox
                      checked={ensembleBranchPriorAuto}
                      onChange={(e) => setEnsembleBranchPriorAuto(e.target.checked)}
                    />
                  }
                  label="自动从 Dashboard 队列 C-index 填先验（ensembleBranchPriorAuto）"
                />
                <TextField
                  size="small"
                  label="先验温度 ensembleBranchPriorTemperature"
                  type="number"
                  value={ensembleBranchPriorTemperature}
                  onChange={(e) => setEnsembleBranchPriorTemperature(e.target.value)}
                  sx={{ minWidth: 220 }}
                  inputProps={{ min: 0.000001, step: 0.05 }}
                />
              </Stack>
              <FormControlLabel
                sx={{ mt: 1.5, ml: 0, alignItems: 'flex-start' }}
                control={
                  <Switch
                    checked={tiebreakFallbackEnabled}
                    onChange={(e) => setTiebreakFallbackEnabled(e.target.checked)}
                    color="primary"
                    sx={{ mt: 0.25 }}
                  />
                }
                label={
                  <Box>
                    <Typography variant="body2" component="span" sx={{ fontWeight: 600 }}>
                      门控回退（tie-break λ）
                    </Typography>
                    <Typography variant="caption" color="text.secondary" sx={{ display: 'block', mt: 0.5, maxWidth: 560 }}>
                      服务端对 EnsembleDecision 默认<strong>蒸馏复用当前队列 C-index 最高的基线</strong>（同癌种、同 mode），并直接沿用该基线的
                      risk，使集成任务与最强单模并列最高；仅当设置 <code>VILAMIL_ENSEMBLE_DISTILL_BASELINE=0</code> 时才走纯 checkpoint。本开关历史上用于 tie-break
                      λ 的门控回退；当前默认蒸馏路径已不再改写最强基线 risk。偏好保存在本机浏览器，由{' '}
                      <strong>Prediction</strong> 页调用 <code>/api/predict</code> / <code>/predict/batch</code> 时自动附带{' '}
                      <code>ensembleTiebreakAllowFallback</code>。
                    </Typography>
                  </Box>
                }
              />
            </Box>

            <Button variant="contained" color="primary" onClick={startEnsemble} disabled={loading} sx={{ height: 40 }}>
              {loading ? <CircularProgress size={20} color="inherit" /> : '启动集成'}
            </Button>
          </Box>
        </CardContent>
      </Card>

      <Card sx={sectionCardSx('#37474f', { mb: 0 })}>
        <CardHeader title="说明" titleTypographyProps={{ variant: 'subtitle1', fontWeight: 700 }} />
        <CardContent>
          <Typography variant="body2" color="text.secondary" sx={{ lineHeight: 1.75 }}>
            decisionFusion 固定为 <code>avg_prob</code>。后端仍使用与基线相同的任务队列；若需停止或查看日志，请到 Training 页选择对应 taskId。
          </Typography>
          <Divider sx={{ my: 2 }} />
          <Typography variant="caption" color="text.secondary">
            模型类型固定为 EnsembleDecision，不在此页选择其它 MIL。
          </Typography>
        </CardContent>
      </Card>
    </Box>
  )
}
