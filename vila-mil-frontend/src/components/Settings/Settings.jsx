import React, { useEffect, useMemo, useState } from 'react'
import {
  Alert,
  Box,
  Button,
  Card,
  CardContent,
  CardHeader,
  Divider,
  Grid,
  Stack,
  TextField,
  Typography,
} from '@mui/material'
import { alpha } from '@mui/material/styles'
import DownloadIcon from '@mui/icons-material/Download'
import { getApiBaseUrl, healthApi, setApiBaseUrl } from '../../services/api'

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

  const save = () => {
    const v = setApiBaseUrl(apiBaseUrl)
    setApiBaseUrlState(v)
    setNotice(`已保存 API BaseURL：${v}（之后所有接口请求都会使用该前缀）`)
  }

  const quickSteps = useMemo(
    () => [
      '1) Data Management：按癌种上传 20× 与 10× 特征 H5，供病例关联或直接预测使用。',
      '2) Clinical：页面分左右两栏——左栏「随访」导入 CSV 或维护 caseId、time、status；右栏「特征与推理」为当前选中病例绑定双尺度 H5，或上传病理图像由后端在线生成特征（上传成功后可保留预览并缩放查看细节）。',
      '3) Training：选择模型与超参启动训练；同一时间仅允许一个训练任务；支持资源预检与空闲超时自动停止。',
      '4) Model Evaluation：选择已完成的 run 查看训练曲线与摘要；可按需做 KM 与 log-rank 分析。',
      '5) Prediction：仅支持按病例（cases.json）推理，使用 Clinical 中该病例已绑定的 20×/10× 特征。Task 下拉仅列出已结束且含 checkpoint 的任务，并按当前病例特征维度做兼容性校验（不匹配项置灰提示）。',
      '6) Settings（本页）：配置 API BaseURL；查看下方服务器路径与后端说明。',
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
          配置 API 地址、查看平台说明与服务器路径；更换域名或排查接口时可优先检查本页。
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
            </CardContent>
          </Card>
        </Grid>

        <Grid item xs={12} md={6}>
          <Card sx={sectionCardSx('#5c6bc0')}>
            <CardHeader title="平台介绍" titleTypographyProps={{ variant: 'subtitle1', fontWeight: 700 }} />
            <CardContent>
              <Typography variant="body2" color="text.secondary" sx={{ lineHeight: 1.8 }}>
                本系统是一个以<strong>多模型集成（EnsembleDecision）</strong>为核心的病理生存风险预测平台：先训练多路 MIL 基线（如 RRTMIL、AMIL、WiKG、DSMIL、S4MIL），再在统一口径下进行集成推理与对比评估。推理输入采用<strong>双尺度（20× / 10×）H5</strong>；Clinical 支持由 WSI 在线生成特征，结果用于科研流程验证。
                <br />
                <br />
                当前版本提供：
                <br />- 集成核心：EnsembleDecision（当前以概率融合为主），支持与 5 个基线模型同屏对比
                <br />- 训练与评估：多模型 k-fold、日志与 checkpoint 管理；曲线面板支持基线与集成过程可视化
                <br />- 集成展示：最优任务对比中可同时查看验证 Loss / 训练 Loss / 验证 AUC 的统一曲线
                <br />- Clinical：随访管理 + 特征绑定分栏设计，支持“从 WSI 生成”（快速预览≈低采样近似，正式预测=TRIDENT 全量）
                <br />- Prediction：按病例读取已绑定特征进行推理，展示风险得分、分层结果与概率分布
                <br />- 兼容性保护：任务下拉自动执行特征维度校验，不匹配任务置灰提示
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
                <br />- 集成结果依赖各基线质量与覆盖范围；建议先保证单模型训练稳定，再观察 EnsembleDecision 的增益。
                <br />- Clinical 中“快速预览”是近似流程（更快），“正式预测”走 TRIDENT 全量流程（更慢但更正式）。
                <br />- Prediction 中 Task 仅显示已完成且含 checkpoint 的任务；界面不展示训练任务 UUID，需要时可将鼠标悬停在某一选项上查看。
                <br />- 更换部署域名或端口时，在本页修改 API BaseURL；与 Nginx 同域反代时建议仍使用 <code>/api</code>。
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
                下载到本地后，按右侧「建议操作」在对应页面导入或上传。双尺度 H5 体积较大，未在此打包；请使用 Data Management 上传自有特征，或使用已完成训练产生的特征。
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
                    ③ 打开 Prediction → 按病例选择该 caseId，并选择已有 checkpoint 的 Task → Predict。
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
                  {Array.isArray(cfg?.notes) && cfg.notes.length > 0 && (
                    <Grid item xs={12}>
                      <Alert severity="info" sx={{ mt: 1 }}>
                        {cfg.notes.map((n, i) => (
                          <div key={i}>{n}</div>
                        ))}
                      </Alert>
                    </Grid>
                  )}
                </Grid>
              )}
            </CardContent>
          </Card>
        </Grid>
      </Grid>
    </Box>
  )
}

