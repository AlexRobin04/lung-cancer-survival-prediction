import React, { useEffect, useMemo, useRef, useState } from 'react'
import {
  Alert,
  Box,
  Button,
  Card,
  CardContent,
  Chip,
  CircularProgress,
  Divider,
  FormControl,
  InputLabel,
  LinearProgress,
  MenuItem,
  Select,
  TextField,
  ToggleButton,
  ToggleButtonGroup,
  Typography,
} from '@mui/material'
import { clinicalApi, dataApi } from '../../services/api'
import useCancerOptions from '../../hooks/useCancerOptions'
import Toast from '../common/Toast.jsx'
import RasterPreview from './RasterPreview.jsx'

function ColumnTitle({ children }) {
  return (
    <Typography
      variant="overline"
      color="text.secondary"
      sx={{
        letterSpacing: 0.8,
        display: 'block',
        mb: 1.25,
        fontWeight: 700,
      }}
    >
      {children}
    </Typography>
  )
}

export default function Clinical() {
  const fileRef = useRef(null)
  /** 上传成功后保留预览（blob URL 由本组件 revoke） */
  const persistedRasterRef = useRef(null)
  const [persistedRasterPreview, setPersistedRasterPreview] = useState(null)
  const [error, setError] = useState('')
  const [notice, setNotice] = useState('')
  const [cases, setCases] = useState([])
  const { cancerOptions, cancer: selectedCancer, setCancer: setSelectedCancer } = useCancerOptions('LUSC')
  const [feature20Files, setFeature20Files] = useState([])
  const [feature10Files, setFeature10Files] = useState([])

  const [selectedCaseId, setSelectedCaseId] = useState('')
  const [newCaseId, setNewCaseId] = useState('')
  const [editTime, setEditTime] = useState('')
  const [editStatus, setEditStatus] = useState('0')
  const [selectedF20, setSelectedF20] = useState('')
  const [selectedF10, setSelectedF10] = useState('')
  const [assocMode, setAssocMode] = useState('h5') // h5 | raster | trident
  const [rasterFile, setRasterFile] = useState(null)
  const [tridentMpp, setTridentMpp] = useState('0.25')
  const [associating, setAssociating] = useState(false)
  const [associateProgress, setAssociateProgress] = useState(0)

  const caseOptions = useMemo(() => cases.map((c) => c.caseId), [cases])
  const selectedCase = useMemo(() => cases.find((c) => c.caseId === selectedCaseId) || null, [cases, selectedCaseId])
  const hasBoundFeatures = Boolean(selectedCase?.feature20FileId && selectedCase?.feature10FileId)

  useEffect(() => {
    persistedRasterRef.current = persistedRasterPreview
  }, [persistedRasterPreview])

  useEffect(() => {
    return () => {
      if (persistedRasterRef.current?.url) {
        URL.revokeObjectURL(persistedRasterRef.current.url)
      }
    }
  }, [])

  useEffect(() => {
    if (!selectedCase) {
      setEditTime('')
      setEditStatus('0')
      return
    }
    setEditTime(selectedCase.time ?? '')
    setEditStatus(String(selectedCase.status ?? 0))
  }, [selectedCase])

  const loadAll = async () => {
    setError('')
    try {
      const [cRes, f20, f10] = await Promise.all([
        clinicalApi.listCases(),
        dataApi.getFeatures(selectedCancer, '20'),
        dataApi.getFeatures(selectedCancer, '10'),
      ])
      setCases(cRes?.cases || [])
      setFeature20Files(f20?.files || [])
      setFeature10Files(f10?.files || [])
    } catch (e) {
      setError(e?.response?.data?.message || e.message || '加载失败')
    }
  }

  useEffect(() => {
    loadAll()
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [selectedCancer])

  const uploadCsv = async () => {
    const f = fileRef.current?.files?.[0]
    if (!f) {
      setError('请先选择 CSV 文件')
      return
    }
    setError('')
    setNotice('')
    try {
      const res = await clinicalApi.uploadCsv(f)
      setNotice(`已导入 ${res?.count ?? '—'} 条病例`)
      await loadAll()
    } catch (e) {
      setError(e?.response?.data?.message || e.message || '导入失败')
    } finally {
      if (fileRef.current) fileRef.current.value = ''
    }
  }

  const createCase = async () => {
    const cid = String(newCaseId || '').trim()
    if (!cid) {
      setError('请先输入 caseId')
      return
    }
    setError('')
    setNotice('')
    try {
      const res = await clinicalApi.createCase({ caseId: cid })
      setNotice(res?.created ? `已新增：${cid}` : `已选中已有病例：${cid}`)
      setSelectedCaseId(cid)
      setNewCaseId('')
      await loadAll()
    } catch (e) {
      setError(e?.response?.data?.message || e.message || '创建病例失败')
    }
  }

  const saveFollowup = async () => {
    if (!selectedCaseId) {
      setError('请先在左侧选择病例')
      return
    }
    setError('')
    setNotice('')
    try {
      await clinicalApi.createCase({
        caseId: selectedCaseId,
        slideId: selectedCase?.slideId || '',
        time: editTime === '' ? 0 : Number(editTime),
        status: Number(editStatus),
      })
      setNotice(`已保存随访：${selectedCaseId}`)
      await loadAll()
    } catch (e) {
      setError(e?.response?.data?.message || e.message || '保存失败')
    }
  }

  const associateFeaturesH5 = async () => {
    if (!selectedCaseId) {
      setError('请先在左侧选择病例')
      return
    }
    if (!selectedF20 || !selectedF10) {
      setError('请同时选择 20× 与 10× 特征文件')
      return
    }
    setError('')
    setNotice('')
    setAssociating(true)
    try {
      await clinicalApi.associateFeatures({
        caseId: selectedCaseId,
        cancer: selectedCancer,
        feature20FileId: selectedF20,
        feature10FileId: selectedF10,
      })
      setNotice('特征已关联到当前病例')
      await loadAll()
    } catch (e) {
      setError(e?.response?.data?.message || e.message || '关联失败')
    } finally {
      setAssociating(false)
    }
  }

  const associateFeaturesFromUpload = async () => {
    if (!selectedCaseId) {
      setError('请先在左侧选择病例')
      return
    }
    if (!rasterFile) {
      setError('请选择病理图像文件')
      return
    }
    setError('')
    setNotice('')
    setAssociating(true)
    setAssociateProgress(2)
    const t0 = Date.now()
    const timer = setInterval(() => {
      const dt = Date.now() - t0
      const target = dt < 8000 ? 35 : dt < 20000 ? 65 : dt < 40000 ? 85 : 95
      setAssociateProgress((p) => (p < target ? p + 2 : p))
    }, 600)
    try {
      const extractor = assocMode === 'trident' ? 'trident' : 'raster'
      if (extractor === 'trident' && rasterFile && /\.(png|jpe?g)$/i.test(rasterFile.name || '') && !(Number(tridentMpp) > 0)) {
        setError('TRIDENT 处理 PNG/JPEG 需填写 mpp（如 0.25）')
        return
      }
      await clinicalApi.associateFeatures({
        caseId: selectedCaseId,
        cancer: selectedCancer,
        file: rasterFile,
        extractor,
        mpp: extractor === 'trident' ? tridentMpp : undefined,
      })
      setNotice(extractor === 'trident' ? '已通过 TRIDENT 生成特征并关联' : '已从图像生成特征并关联')
      setAssociateProgress(100)
      if (rasterFile) {
        setPersistedRasterPreview((prev) => {
          if (prev?.url) URL.revokeObjectURL(prev.url)
          return { url: URL.createObjectURL(rasterFile), name: rasterFile.name }
        })
      }
      setRasterFile(null)
      await loadAll()
    } catch (e) {
      setError(e?.response?.data?.message || e.message || '关联失败')
    } finally {
      clearInterval(timer)
      setAssociating(false)
      setTimeout(() => setAssociateProgress(0), 800)
    }
  }

  const deleteCase = async () => {
    if (!selectedCaseId) {
      setError('请先在左侧选择病例')
      return
    }
    const ok = window.confirm(`删除病例「${selectedCaseId}」？将清除该病例下的特征关联。`)
    if (!ok) return
    setError('')
    setNotice('')
    try {
      await clinicalApi.deleteCase(selectedCaseId)
      setNotice(`已删除：${selectedCaseId}`)
      setSelectedCaseId('')
      setSelectedF20('')
      setSelectedF10('')
      setPersistedRasterPreview((prev) => {
        if (prev?.url) URL.revokeObjectURL(prev.url)
        return null
      })
      setRasterFile(null)
      await loadAll()
    } catch (e) {
      setError(e?.response?.data?.message || e.message || '删除失败')
    }
  }

  return (
    <Box sx={{ mt: 2 }}>
      <Box
        sx={(theme) => ({
          mb: 3,
          p: { xs: 2, sm: 2.5 },
          borderRadius: 2,
          border: '1px solid',
          borderColor: 'divider',
          background:
            theme.palette.mode === 'dark'
              ? 'linear-gradient(115deg, rgba(25,118,210,0.20) 0%, rgba(2,136,209,0.08) 100%)'
              : 'linear-gradient(115deg, rgba(25,118,210,0.12) 0%, rgba(2,136,209,0.05) 100%)',
        })}
      >
        <Typography variant="h4" gutterBottom sx={{ fontWeight: 700, mb: 1 }}>
          Clinical
        </Typography>
        <Typography variant="body2" color="text.secondary" sx={{ maxWidth: 760 }}>
          左栏维护随访与病例；右栏为当前选中病例准备推理用的双尺度特征。两栏共用同一「当前病例」。
        </Typography>
      </Box>

      {error && (
        <Alert severity="error" sx={{ mb: 2 }} onClose={() => setError('')}>
          {error}
        </Alert>
      )}
      <Toast open={!!notice} message={notice} severity="success" onClose={() => setNotice('')} />

      <Box
        sx={{
          display: 'grid',
          gridTemplateColumns: { xs: '1fr', md: 'minmax(0, 1fr) minmax(0, 1fr)' },
          gap: { xs: 2, md: 3 },
          alignItems: 'stretch',
        }}
      >
        {/* —— 随访 —— */}
        <Card
          variant="outlined"
          sx={(theme) => ({
            borderRadius: 2,
            borderColor: 'divider',
            bgcolor: theme.palette.mode === 'dark' ? 'action.selected' : 'grey.50',
            boxShadow: theme.palette.mode === 'dark' ? 'none' : '0 6px 18px rgba(15,23,42,0.06)',
            transition: 'box-shadow .2s ease, transform .2s ease',
            '&:hover': {
              boxShadow: theme.palette.mode === 'dark' ? 'none' : '0 10px 24px rgba(15,23,42,0.10)',
              transform: 'translateY(-1px)',
            },
          })}
        >
          <CardContent sx={{ p: { xs: 2, sm: 2.5 } }}>
            <ColumnTitle>随访</ColumnTitle>
            <Typography variant="subtitle1" sx={{ fontWeight: 600, mb: 0.5 }}>
              病例与生存数据
            </Typography>
            <Typography variant="body2" color="text.secondary" sx={{ mb: 2 }}>
              导入 CSV 或手动新增 caseId，再编辑 time / status（用于随访与部分分析）。
            </Typography>

            <Divider sx={{ mb: 2 }} />

            <Typography variant="caption" color="text.secondary" sx={{ display: 'block', mb: 1 }}>
              批量导入
            </Typography>
            <Alert severity="info" sx={{ mb: 1.5, py: 0.5 }} icon={false}>
              CSV 需含 <code>case_id</code>、<code>time</code>、<code>status</code>（1=事件，0=删失）；其余列进入{' '}
              <code>clinicalVars</code>。
            </Alert>
            <Box sx={{ display: 'flex', flexWrap: 'wrap', gap: 1, mb: 2 }}>
              <Button component="label" variant="outlined" size="small">
                选择 CSV
                <input ref={fileRef} type="file" accept=".csv" hidden />
              </Button>
              <Button variant="contained" size="small" onClick={uploadCsv}>
                导入
              </Button>
            </Box>

            <Typography variant="caption" color="text.secondary" sx={{ display: 'block', mb: 1 }}>
              病例操作
            </Typography>
            <Box
              sx={{
                border: '1px solid',
                borderColor: 'divider',
                borderRadius: 1.5,
                bgcolor: 'background.paper',
                p: 1.5,
                mb: 1.5,
                boxShadow: 'inset 0 1px 0 rgba(255,255,255,0.04)',
              }}
            >
              <Typography variant="subtitle2" sx={{ fontWeight: 600, mb: 1 }}>
                A. 新增病例
              </Typography>
              <Box sx={{ display: 'flex', flexWrap: 'wrap', gap: 1, alignItems: 'center' }}>
                <TextField
                  size="small"
                  label="新 caseId"
                  value={newCaseId}
                  onChange={(e) => setNewCaseId(e.target.value)}
                  sx={{ minWidth: 200 }}
                  placeholder="如 CASE_1001"
                />
                <Button variant="outlined" size="small" onClick={createCase}>
                  新增并选中
                </Button>
              </Box>
            </Box>

            <Box
              sx={{
                border: '1px solid',
                borderColor: 'divider',
                borderRadius: 1.5,
                bgcolor: 'background.paper',
                p: 1.5,
                boxShadow: 'inset 0 1px 0 rgba(255,255,255,0.04)',
              }}
            >
              <Typography variant="subtitle2" sx={{ fontWeight: 600, mb: 1 }}>
                B. 已有病例管理（选择 / 修改 / 删除）
              </Typography>
              <FormControl fullWidth size="small" sx={{ mb: 1.5 }}>
                <InputLabel id="caseid-edit-label">当前病例</InputLabel>
                <Select
                  labelId="caseid-edit-label"
                  label="当前病例"
                  value={selectedCaseId}
                  onChange={(e) => setSelectedCaseId(e.target.value)}
                >
                  {caseOptions.map((cid) => (
                    <MenuItem key={cid} value={cid}>
                      {cid}
                    </MenuItem>
                  ))}
                </Select>
              </FormControl>

              <Box sx={{ display: 'flex', flexWrap: 'wrap', gap: 1, alignItems: 'center', mb: 1.5 }}>
                <TextField
                  size="small"
                  label="time"
                  type="number"
                  value={editTime}
                  onChange={(e) => setEditTime(e.target.value)}
                  sx={{ width: 120 }}
                  helperText="随访时间"
                />
                <FormControl size="small" sx={{ minWidth: 140 }}>
                  <InputLabel id="status-edit-label">status</InputLabel>
                  <Select
                    labelId="status-edit-label"
                    label="status"
                    value={editStatus}
                    onChange={(e) => setEditStatus(e.target.value)}
                  >
                    <MenuItem value="0">0 删失</MenuItem>
                    <MenuItem value="1">1 事件</MenuItem>
                  </Select>
                </FormControl>
              </Box>

              <Box sx={{ display: 'flex', flexWrap: 'wrap', gap: 1 }}>
                <Button variant="contained" size="small" onClick={saveFollowup} disabled={!selectedCaseId}>
                  保存随访
                </Button>
                <Button variant="outlined" color="error" size="small" onClick={deleteCase} disabled={!selectedCaseId}>
                  删除病例
                </Button>
              </Box>
            </Box>

            <Typography variant="caption" color="text.secondary" sx={{ mt: 2, display: 'block' }}>
              共 {cases.length} 个病例
            </Typography>
          </CardContent>
        </Card>

        {/* —— 特征与推理 —— */}
        <Card
          variant="outlined"
          sx={(theme) => ({
            borderRadius: 2,
            borderColor: 'divider',
            bgcolor: theme.palette.mode === 'dark' ? 'background.paper' : '#fcfdff',
            boxShadow: theme.palette.mode === 'dark' ? 'none' : '0 6px 18px rgba(15,23,42,0.06)',
            transition: 'box-shadow .2s ease, transform .2s ease',
            '&:hover': {
              boxShadow: theme.palette.mode === 'dark' ? 'none' : '0 10px 24px rgba(15,23,42,0.10)',
              transform: 'translateY(-1px)',
            },
          })}
        >
          <CardContent sx={{ p: { xs: 2, sm: 2.5 } }}>
            <ColumnTitle>特征与推理</ColumnTitle>
            <Typography variant="subtitle1" sx={{ fontWeight: 600, mb: 0.5 }}>
              绑定 MIL 所需双尺度 H5
            </Typography>
            <Typography variant="body2" color="text.secondary" sx={{ mb: 2 }}>
              推理只读 20× 与 10× 特征。可选：在数据管理页已上传的 H5，或在此上传图像/WSI 由后端生成特征（ResNet/TRIDENT）。
            </Typography>

            <Divider sx={{ mb: 2 }} />

            {!selectedCaseId ? (
              <Alert severity="warning" sx={{ mb: 2 }}>
                请先在<strong>左栏</strong>选择或新增一个病例，再在此关联特征。
              </Alert>
            ) : (
              <Box sx={{ mb: 2, display: 'flex', alignItems: 'center', gap: 1, flexWrap: 'wrap' }}>
                <Typography variant="body2" color="text.secondary">
                  操作对象
                </Typography>
                <Chip size="small" label={selectedCaseId} color="primary" variant="outlined" />
              </Box>
            )}

            <Typography variant="caption" color="text.secondary" sx={{ display: 'block', mb: 1 }}>
              方式
            </Typography>
            <ToggleButtonGroup
              value={assocMode}
              exclusive
              size="small"
              fullWidth
              sx={{ mb: 2 }}
              onChange={(_, v) => v && setAssocMode(v)}
              disabled={!selectedCaseId || associating}
            >
              <ToggleButton value="h5">已有 H5 文件</ToggleButton>
              <ToggleButton value="raster">从图像生成（ResNet）</ToggleButton>
              <ToggleButton value="trident">从 WSI 生成（TRIDENT）</ToggleButton>
            </ToggleButtonGroup>

            {assocMode === 'h5' && (
              <Box sx={{ display: 'flex', flexDirection: 'column', gap: 1.5 }}>
                <FormControl fullWidth size="small">
                  <InputLabel id="feat-cancer-label">癌种</InputLabel>
                  <Select
                    labelId="feat-cancer-label"
                    label="癌种"
                    value={selectedCancer}
                    onChange={(e) => setSelectedCancer(e.target.value)}
                  >
                    {cancerOptions.map((opt) => (
                      <MenuItem key={opt.value} value={opt.value}>
                        {opt.label}
                      </MenuItem>
                    ))}
                  </Select>
                </FormControl>
                <FormControl fullWidth size="small">
                  <InputLabel id="f20-label">20× 特征</InputLabel>
                  <Select
                    labelId="f20-label"
                    label="20× 特征"
                    value={selectedF20}
                    onChange={(e) => setSelectedF20(e.target.value)}
                  >
                    {feature20Files.map((f) => (
                      <MenuItem key={f.id} value={f.id}>
                        {f.name || f.id}
                      </MenuItem>
                    ))}
                  </Select>
                </FormControl>
                <FormControl fullWidth size="small">
                  <InputLabel id="f10-label">10× 特征</InputLabel>
                  <Select
                    labelId="f10-label"
                    label="10× 特征"
                    value={selectedF10}
                    onChange={(e) => setSelectedF10(e.target.value)}
                  >
                    {feature10Files.map((f) => (
                      <MenuItem key={f.id} value={f.id}>
                        {f.name || f.id}
                      </MenuItem>
                    ))}
                  </Select>
                </FormControl>
                <Button
                  variant="contained"
                  onClick={associateFeaturesH5}
                  disabled={!selectedCaseId || associating || !selectedF20 || !selectedF10}
                >
                  {associating ? <CircularProgress size={20} color="inherit" /> : '保存关联'}
                </Button>
              </Box>
            )}

            {(assocMode === 'raster' || assocMode === 'trident') && (
              <Box sx={{ display: 'flex', flexDirection: 'column', gap: 1.5 }}>
                {associating && (
                  <Box sx={{ mb: 0.5 }}>
                    <Typography variant="caption" color="text.secondary" sx={{ display: 'block', mb: 0.6 }}>
                      正在生成双尺度特征并写入病例，请稍候（{associateProgress}%）
                    </Typography>
                    <LinearProgress variant="determinate" value={Math.max(2, Math.min(100, associateProgress))} />
                  </Box>
                )}
                {assocMode === 'trident' ? (
                  <Alert severity="info" sx={{ py: 0.5 }}>
                    使用 TRIDENT 提取双尺度 H5（20x/10x）。建议上传 WSI（如 .svs/.ndpi/.mrxs/.scn/.tiff）。
                  </Alert>
                ) : (
                  <Alert severity="warning" sx={{ py: 0.5 }}>
                    ResNet 在线特征路径仅用于流程验证，与标准病理特征分布可能不一致。
                  </Alert>
                )}
                <FormControl fullWidth size="small">
                  <InputLabel id="raster-cancer-label">癌种</InputLabel>
                  <Select
                    labelId="raster-cancer-label"
                    label="癌种"
                    value={selectedCancer}
                    onChange={(e) => setSelectedCancer(e.target.value)}
                  >
                    {cancerOptions.map((opt) => (
                      <MenuItem key={opt.value} value={opt.value}>
                        {opt.label}
                      </MenuItem>
                    ))}
                  </Select>
                </FormControl>
                {assocMode === 'trident' && (
                  <TextField
                    size="small"
                    label="MPP（可选，PNG/JPEG 必填）"
                    value={tridentMpp}
                    onChange={(e) => setTridentMpp(e.target.value)}
                    placeholder="如 0.25"
                    helperText="WSI 通常可从元数据读取；普通图片无元数据时请填写"
                  />
                )}
                <Button component="label" variant="outlined" disabled={!selectedCaseId || associating}>
                  {assocMode === 'trident' ? '选择 WSI / 图像文件' : '选择图像（PNG / JPEG …）'}
                  <input
                    type="file"
                    hidden
                    accept={
                      assocMode === 'trident'
                        ? '.svs,.ndpi,.mrxs,.scn,.tif,.tiff,image/png,image/jpeg'
                        : 'image/png,image/jpeg,image/webp,image/bmp,.tif,.tiff'
                    }
                    onChange={(e) => {
                      const f = e.target.files?.[0] || null
                      setPersistedRasterPreview((prev) => {
                        if (prev?.url) URL.revokeObjectURL(prev.url)
                        return null
                      })
                      setRasterFile(f)
                    }}
                  />
                </Button>
                {!rasterFile && !persistedRasterPreview && (
                  <Typography variant="body2" color="text.secondary">
                    未选择文件
                  </Typography>
                )}
                <RasterPreview file={rasterFile} persisted={persistedRasterPreview} />
                <Button
                  variant="contained"
                  onClick={associateFeaturesFromUpload}
                  disabled={!selectedCaseId || associating || !rasterFile}
                >
                  {associating ? <CircularProgress size={20} color="inherit" /> : assocMode === 'trident' ? '上传并用 TRIDENT 生成' : '上传并生成特征'}
                </Button>
              </Box>
            )}

            <Divider sx={{ my: 2 }} />

            {hasBoundFeatures ? (
              <>
                <Typography variant="caption" color="text.secondary" sx={{ display: 'block', mb: 0.5 }}>
                  当前病例特征状态
                </Typography>
                <Box
                  sx={(theme) => ({
                    px: 1.25,
                    py: 1,
                    borderRadius: 1.2,
                    border: '1px solid',
                    borderColor: theme.palette.mode === 'dark' ? 'divider' : 'rgba(25,118,210,0.25)',
                    bgcolor: theme.palette.mode === 'dark' ? 'action.hover' : 'rgba(25,118,210,0.04)',
                  })}
                >
                  <Typography variant="body2" color="text.secondary" component="div" sx={{ lineHeight: 1.7 }}>
                    来源：
                    {selectedCase?.featureSource === 'raster_derived'
                      ? '图像派生'
                      : selectedCase?.featureSource === 'trident_derived'
                        ? 'TRIDENT 派生'
                      : selectedCase?.featureSource === 'h5_pair'
                        ? '已选 H5'
                      : selectedCase?.feature20FileId
                        ? '已登记'
                        : '未关联'}
                    <br />
                    20×：{selectedCase?.feature20FileId ? <code>{selectedCase.feature20FileId}</code> : '—'}
                    <br />
                    10×：{selectedCase?.feature10FileId ? <code>{selectedCase.feature10FileId}</code> : '—'}
                    {selectedCase?.rasterSourceFileName ? (
                      <>
                        <br />
                        图像文件名：{selectedCase.rasterSourceFileName}
                      </>
                    ) : null}
                  </Typography>
                </Box>
              </>
            ) : (
              <Typography
                variant="body2"
                color="text.secondary"
                sx={{
                  px: 1.25,
                  py: 1,
                  borderRadius: 1.2,
                  bgcolor: 'action.hover',
                  border: '1px dashed',
                  borderColor: 'divider',
                }}
              >
                绑定成功后将在此显示当前病例的详细特征信息。
              </Typography>
            )}
          </CardContent>
        </Card>
      </Box>
    </Box>
  )
}
