import React, { useEffect, useMemo, useState } from 'react'
import {
  Alert,
  Box,
  Button,
  Card,
  CardContent,
  CardHeader,
  Divider,
  FormControlLabel,
  Grid,
  Stack,
  Switch,
  TextField,
  Typography,
} from '@mui/material'
import { alpha } from '@mui/material/styles'
import DownloadIcon from '@mui/icons-material/Download'
import { getApiBaseUrl, healthApi, setApiBaseUrl } from '../../services/api'
import {
  readClinicalDemoFakeExtraction,
  writeClinicalDemoFakeExtraction,
} from '../../utils/clinicalDemoStorage'
import {
  readEnsembleTiebreakAllowFallback,
  writeEnsembleTiebreakAllowFallback,
} from '../../constants/ensemblePredictPrefs'

/** 静态样例目录（Vite public/test-samples，部署后与站点同域） */
function testSampleUrl(name) {
  const base = import.meta.env.BASE_URL || '/'
  const prefix = base.endsWith('/') ? base : `${base}/`
  return `${prefix}test-samples/${name}`
}

const sectionCardSx = (accent) => (theme) => ({
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

export default function Settings() {
  const [error, setError] = useState('')
  const [notice, setNotice] = useState('')
  const [apiBaseUrl, setApiBaseUrlState] = useState(getApiBaseUrl())
  const [cfg, setCfg] = useState(null)
  const [demoFakeExtraction, setDemoFakeExtraction] = useState(() =>
    typeof window !== 'undefined' ? readClinicalDemoFakeExtraction() : false
  )
  const [tiebreakFallbackEnabled, setTiebreakFallbackEnabled] = useState(() =>
    typeof window !== 'undefined' ? readEnsembleTiebreakAllowFallback() : true
  )

  const loadCfg = async () => {
    setError('')
    try {
      const c = await healthApi.config()
      setCfg(c)
    } catch (e) {
      setError(e?.response?.data?.message || e.message || '加载服务器配置失败（/api/config）')
      setCfg(null)
    }
  }

  useEffect(() => {
    loadCfg()
  }, [])

  useEffect(() => {
    writeEnsembleTiebreakAllowFallback(tiebreakFallbackEnabled)
  }, [tiebreakFallbackEnabled])

  const save = () => {
    const v = setApiBaseUrl(apiBaseUrl)
    setApiBaseUrlState(v)
    setNotice(`已保存 API BaseURL：${v}（之后所有接口请求都会使用该前缀）`)
  }

  const quickSteps = useMemo(
    () => [
      '1) Clinical：左栏随访（CSV 导入带状态提示、病例维护）；左栏底部展示「当前病例特征状态」；右栏 WSI 选择与上传；绑定后「已绑定病例图片预览」在右栏，说明文字可收起到 ℹ️。',
      '2) Training：仅配置「单模型 MIL」（RRTMIL / AMIL / WiKG / DSMIL / S4MIL）与超参；与集成任务「共用队列」；单任务并发；可看日志与停止。',
      '3) Ensemble：独立页的 EnsembleDecision（分支纳入/排除、先验与温度等）；仍走 POST /api/training/start，队列与日志在 Training 查看。',
      '4) Evaluation：最优任务曲线对比（多模型 Loss/AUC）、评估 runs 与指标总览；与 Dashboard 部分统计同源。',
      '5) Prediction：按病例与 Task 推理/批量推理；历史与「队列 C-index（按模型）」；请求可携带「门控回退」偏好（本页或 Ensemble 页同步）。',
      '6) Dashboard：总览近期训练、预测与评估；可刷新聚合。',
      '7) Settings（本页）：API BaseURL；服务器只读路径；「演示」开关；与 Ensemble 联动的 tie-break 门控回退偏好。',
    ],
    []
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
              ? 'linear-gradient(115deg, rgba(92,107,192,0.24) 0%, rgba(25,118,210,0.12) 100%)'
              : 'linear-gradient(115deg, rgba(92,107,192,0.12) 0%, rgba(25,118,210,0.06) 100%)',
        })}
      >
        <Typography variant="h4" gutterBottom sx={{ fontWeight: 700, mb: 1 }}>
          设置
        </Typography>
        <Typography variant="body2" color="text.secondary">
          配置 API 地址、浏览器本地偏好；查看平台说明与服务器只读路径。更换域名或排查接口时可优先检查本页。
        </Typography>
      </Box>

      {error && (
        <Alert severity="error" sx={{ mb: 2 }} onClose={() => setError('')}>
          {error}
        </Alert>
      )}
      {notice && (
        <Alert severity="success" sx={{ mb: 2 }} onClose={() => setNotice('')}>
          {notice}
        </Alert>
      )}

      <Grid container spacing={3}>
        <Grid item xs={12} md={6}>
          <Card sx={sectionCardSx('#1976d2')}>
            <CardHeader
              title="接口与部署"
              subheader="通常保持 /api（Nginx 反代）。"
              titleTypographyProps={{ variant: 'subtitle1', fontWeight: 700 }}
            />
            <CardContent>
              <TextField
                fullWidth
                label="API BaseURL"
                value={apiBaseUrl}
                onChange={(e) => setApiBaseUrlState(e.target.value)}
                helperText="示例：/api 或 http://121.41.39.63/api"
              />
              <Box sx={{ display: 'flex', gap: 1, mt: 2, flexWrap: 'wrap' }}>
                <Button variant="contained" onClick={save}>
                  保存
                </Button>
                <Button variant="outlined" onClick={loadCfg}>
                  重新读取服务器路径
                </Button>
              </Box>
              <Alert
                severity="info"
                sx={(theme) => ({
                  mt: 2,
                  borderRadius: 2,
                  border: '1px solid',
                  borderColor: alpha(theme.palette.info.main, 0.35),
                  bgcolor: alpha(theme.palette.info.main, theme.palette.mode === 'dark' ? 0.12 : 0.06),
                })}
              >
                如果你使用 Nginx（推荐），前端访问域名即可，接口保持 <code>/api</code>。
                <br />
                若前端不走同域名（跨域），可以把 BaseURL 改成带域名的完整地址。
              </Alert>
              <Divider sx={{ my: 2 }} />
              <Typography variant="subtitle2" sx={{ fontWeight: 700, mb: 1 }}>
                浏览器本地偏好（与 Ensemble / Prediction 联动）
              </Typography>
              <FormControlLabel
                control={
                  <Switch
                    checked={tiebreakFallbackEnabled}
                    onChange={(e) => setTiebreakFallbackEnabled(e.target.checked)}
                    color="primary"
                  />
                }
                label={
                  <Box>
                    <Typography variant="body2">门控回退（蒸馏 tie-break 的 λ）</Typography>
                    <Typography variant="caption" color="text.secondary" sx={{ display: 'block', mt: 0.5 }}>
                      与 <strong>Ensemble</strong> 页开关写入同一本地项；<strong>Prediction</strong> 发起{' '}
                      <code>/api/predict</code> 时会附带 <code>ensembleTiebreakAllowFallback</code>。开启=验证不足或全量退化时可将
                      λ 置 0（稳健）；关闭=始终用学习到的 λ。服务端蒸馏路径需环境变量{' '}
                      <code>VILAMIL_ENSEMBLE_DISTILL_BASELINE=1</code> 才生效。
                    </Typography>
                  </Box>
                }
                sx={{ alignItems: 'flex-start', ml: 0 }}
              />
            </CardContent>
          </Card>
        </Grid>

        <Grid item xs={12} md={6}>
          <Card sx={sectionCardSx('#5c6bc0')}>
            <CardHeader title="平台介绍" titleTypographyProps={{ variant: 'subtitle1', fontWeight: 700 }} />
            <CardContent>
              <Typography variant="body2" color="text.secondary" sx={{ lineHeight: 1.8 }}>
                本系统面向<strong>基于病理图像/特征的肺癌生存风险预测</strong>：先完成多路 MIL 基线（RRTMIL、AMIL、WiKG、DSMIL、S4MIL），再在统一数据口径下做<strong>决策级 EnsembleDecision</strong> 与对比评估。推理输入为<strong>双尺度（20× / 10×）H5</strong>；Clinical 支持由 WSI 在线生成特征并绑定病例，用于科研与流程验证。
                <br />
                <br />
                当前版本提供：
                <br />- <strong>Training</strong>：仅单模型 MIL 训练与队列/日志；<strong>Ensemble</strong>：独立页的 EnsembleDecision 配置（与基线<strong>共用</strong>训练队列与 <code>/api/training/start</code>）
                <br />- <strong>Evaluation</strong>：多模型最优任务曲线（Loss/AUC）、评估 runs；<strong>Dashboard</strong>：总览近期训练、预测与评估
                <br />- <strong>Clinical</strong>：左栏随访与「当前病例特征状态」；右栏 WSI 与特征绑定；预览说明可收起到 ℹ️；CSV 导入带状态提示
                <br />- <strong>Prediction</strong>：单例/批量推理、历史记录与<strong>队列 C-index（按模型）</strong>；可与服务端蒸馏/tie-break 策略配合（见本页本地偏好）
                <br />- <strong>Settings</strong>：API BaseURL、服务器路径只读、演示开关、与 Ensemble 同步的 tie-break 门控回退偏好
              </Typography>
              <Divider sx={{ my: 2 }} />
              <Typography
                variant="caption"
                color="text.secondary"
                component="div"
                sx={(theme) => ({
                  lineHeight: 1.7,
                  p: 1.5,
                  borderRadius: 1.5,
                  bgcolor: theme.palette.mode === 'dark' ? 'action.hover' : alpha('#5c6bc0', 0.06),
                  border: '1px solid',
                  borderColor: 'divider',
                })}
              >
                <strong>使用前请知悉：</strong>
                <br />- 集成结果依赖各基线质量与覆盖范围；建议先完成五路基线再排 EnsembleDecision。
                <br />- Clinical「快速预览」为低采样近似；「正式预测」为 TRIDENT 全量，耗时更长。
                <br />- Prediction 的 Task 仅列出已完成且含 checkpoint 的任务；队列 C-index 依赖 Clinical 随访与预测历史齐全。
                <br />- 自动分支先验（Dashboard 同源 C-index）在集成任务提交时由后端计算；预测量大时后端已做候选任务过滤与缓存以减轻卡顿。
                <br />- 更换部署域名时在本页修改 API BaseURL；与 Nginx 同域反代时建议仍使用 <code>/api</code>。
              </Typography>
            </CardContent>
          </Card>
        </Grid>

        <Grid item xs={12}>
          <Card sx={sectionCardSx('#00897b')}>
            <CardHeader
              title="使用方法（推荐流程）"
              titleTypographyProps={{ variant: 'subtitle1', fontWeight: 700 }}
            />
            <CardContent>
              <Stack spacing={1.25}>
                {quickSteps.map((t, i) => (
                  <Box
                    key={t}
                    sx={(theme) => ({
                      display: 'flex',
                      gap: 1.5,
                      alignItems: 'flex-start',
                      p: 1.25,
                      borderRadius: 1.5,
                      border: '1px solid',
                      borderColor: 'divider',
                      bgcolor: theme.palette.mode === 'dark' ? 'action.hover' : alpha('#00897b', 0.04),
                    })}
                  >
                    <Typography
                      variant="caption"
                      sx={{
                        flexShrink: 0,
                        fontWeight: 800,
                        color: '#00897b',
                        minWidth: 22,
                        mt: 0.25,
                      }}
                    >
                      {i + 1}
                    </Typography>
                    <Typography variant="body2" sx={{ lineHeight: 1.65 }}>
                      {t.replace(/^\d+\)\s*/, '')}
                    </Typography>
                  </Box>
                ))}
              </Stack>
            </CardContent>
          </Card>
        </Grid>

        <Grid item xs={12}>
          <Card sx={sectionCardSx('#ed6c02')}>
            <CardHeader
              title="流程测试样例（\(^o^)/~）"
              subheader="虚构数据，仅供走通 Clinical / Prediction 等流程，不代表真实病例。"
              titleTypographyProps={{ variant: 'subtitle1', fontWeight: 700 }}
            />
            <CardContent>
              <Typography variant="body2" color="text.secondary" sx={{ mb: 2 }}>
                下载到本地后，按右侧「建议操作」在 Clinical 导入随访；双尺度特征请在 Clinical 右栏上传 WSI 由后端生成并绑定病例。
              </Typography>
              <Stack spacing={2.5}>
                <Box
                  sx={(theme) => ({
                    p: 1.5,
                    borderRadius: 1.5,
                    border: '1px solid',
                    borderColor: 'divider',
                    bgcolor: theme.palette.mode === 'dark' ? 'action.hover' : alpha('#ed6c02', 0.04),
                  })}
                >
                  <Typography variant="subtitle2" gutterBottom sx={{ fontWeight: 700 }}>
                    随访 CSV
                  </Typography>
                  <Typography variant="body2" color="text.secondary" sx={{ mb: 1 }}>
                    含 <code>case_id</code>、<code>time</code>、<code>status</code>（0=删失，1=事件）；列{' '}
                    <code>group_label</code> 会进入 <code>clinicalVars</code>。可用于 Clinical 左栏「导入」。
                  </Typography>
                  <Button
                    component="a"
                    href={testSampleUrl('sample_clinical_followup.csv')}
                    download="sample_clinical_followup.csv"
                    variant="outlined"
                    size="small"
                    startIcon={<DownloadIcon />}
                  >
                    下载 sample_clinical_followup.csv
                  </Button>
                </Box>
                <Divider flexItem />
                <Box
                  sx={(theme) => ({
                    p: 1.5,
                    borderRadius: 1.5,
                    border: '1px solid',
                    borderColor: 'divider',
                    bgcolor: theme.palette.mode === 'dark' ? 'action.hover' : alpha('#ed6c02', 0.04),
                  })}
                >
                  <Typography variant="subtitle2" gutterBottom sx={{ fontWeight: 700 }}>
                    病理示例 WSI（SVS）
                  </Typography>
                  <Typography variant="body2" color="text.secondary" sx={{ mb: 1 }}>
                    TCGA-LUSC 示例切片（.svs），用于 Clinical 右栏「从 WSI 生成」联调与快速/正式模式演示。
                  </Typography>
                  <Button
                    component="a"
                    href={testSampleUrl('sample_tcga_lusc_demo.svs')}
                    download="sample_tcga_lusc_demo.svs"
                    variant="outlined"
                    size="small"
                    startIcon={<DownloadIcon />}
                  >
                    下载 sample_tcga_lusc_demo.svs
                  </Button>
                </Box>
                <Divider flexItem />
                <Box
                  sx={(theme) => ({
                    p: 1.5,
                    borderRadius: 1.5,
                    border: '1px solid',
                    borderColor: 'divider',
                    bgcolor: theme.palette.mode === 'dark' ? 'action.hover' : alpha('#ed6c02', 0.04),
                  })}
                >
                  <Typography variant="subtitle2" gutterBottom sx={{ fontWeight: 700 }}>
                    建议操作顺序（可变通）
                  </Typography>
                  <Typography variant="body2" color="text.secondary" component="div" sx={{ lineHeight: 1.75 }}>
                    ① 下载 CSV → Clinical 左栏导入 → 选中或新建与 CSV 一致的 caseId（如 DEMO_FLOW_001）。
                    <br />
                    ② 下载 SVS → Clinical 右栏选「从 WSI 生成」→ 上传 SVS → 选择「快速预览」或「正式预测」。
                    <br />
                    ③（可选）在 Training 完成基线后，打开 Ensemble 配置并启动 EnsembleDecision；队列与日志仍在 Training 查看。
                    <br />
                    ④ 打开 Prediction → 按病例选择该 caseId，并选择已有 checkpoint 的 Task → Predict。
                  </Typography>
                </Box>
              </Stack>
            </CardContent>
          </Card>
        </Grid>

        <Grid item xs={12}>
          <Card sx={sectionCardSx('#455a64')}>
            <CardHeader
              title="服务器路径（只读）"
              subheader="来自后端 /api/config，用于排错与写论文/任务书说明。"
              titleTypographyProps={{ variant: 'subtitle1', fontWeight: 700 }}
            />
            <CardContent>
              {!cfg ? (
                <Alert severity="info">暂无数据（可能后端未更新或 /api/config 不可用）。</Alert>
              ) : (
                <Grid
                  container
                  spacing={1.5}
                  sx={(theme) => ({
                    p: 1.5,
                    borderRadius: 1.5,
                    bgcolor: theme.palette.mode === 'dark' ? 'action.hover' : alpha('#455a64', 0.05),
                    border: '1px solid',
                    borderColor: 'divider',
                  })}
                >
                  {Object.entries(cfg)
                    .filter(([k]) => !['notes'].includes(k))
                    .map(([k, v]) => (
                      <Grid item xs={12} md={6} key={k}>
                        <Typography variant="body2" sx={{ wordBreak: 'break-all' }}>
                          <b>{k}</b>：<code>{String(v)}</code>
                        </Typography>
                      </Grid>
                    ))}
                  <Grid item xs={12}>
                    <Alert
                      severity="info"
                      sx={{
                        mt: 1,
                        display: 'flex',
                        alignItems: 'center',
                        justifyContent: 'space-between',
                        gap: 2,
                        flexWrap: 'wrap',
                      }}
                    >
                      <Box sx={{ flex: '1 1 240px', minWidth: 0 }}>
                        {Array.isArray(cfg?.notes) && cfg.notes.length > 0
                          ? cfg.notes.map((n, i) => (
                              <div key={i}>{n}</div>
                            ))
                          : null}
                      </Box>
                      <FormControlLabel
                        control={
                          <Switch
                            checked={demoFakeExtraction}
                            onChange={(_, checked) => {
                              setDemoFakeExtraction(checked)
                              writeClinicalDemoFakeExtraction(checked)
                            }}
                            color="primary"
                            inputProps={{ 'aria-label': 'Clinical 演示' }}
                          />
                        }
                        label="演示"
                        sx={{ mr: 0, flexShrink: 0 }}
                      />
                    </Alert>
                  </Grid>
                </Grid>
              )}
            </CardContent>
          </Card>
        </Grid>
      </Grid>
    </Box>
  )
}

