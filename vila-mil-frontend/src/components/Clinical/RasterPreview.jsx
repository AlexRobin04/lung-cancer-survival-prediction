import React, { useEffect, useRef, useState } from 'react'
import { Box, IconButton, Slider, Stack, Typography } from '@mui/material'
import ZoomInIcon from '@mui/icons-material/ZoomIn'
import ZoomOutIcon from '@mui/icons-material/ZoomOut'
import RestartAltIcon from '@mui/icons-material/RestartAlt'

const ZOOM_MIN = 0.25
const ZOOM_MAX = 4
const ZOOM_STEP = 0.05

/**
 * 本地病理图像预览：固定外框 + 内部可滚动，支持滑块/按钮缩放。
 * @param {File | null} file - 当前选中的文件（优先显示）
 * @param {{ url: string, name: string } | null} persisted - 上传成功后保留的预览（parent 持有 url 并负责 revoke）
 * @param {{ url: string, name: string } | null} generated - 针对 WSI 生成的缩略图预览（parent 持有 url 并负责 revoke）
 */
export default function RasterPreview({ file, persisted = null, generated = null }) {
  const [fileObjectUrl, setFileObjectUrl] = useState(null)
  const [zoom, setZoom] = useState(1)
  const scrollRef = useRef(null)

  useEffect(() => {
    if (!file) {
      setFileObjectUrl(null)
      return undefined
    }
    const url = URL.createObjectURL(file)
    setFileObjectUrl(url)
    return () => URL.revokeObjectURL(url)
  }, [file])

  const displayUrl = generated?.url || fileObjectUrl || persisted?.url || null
  const displayName = generated?.name ?? file?.name ?? persisted?.name ?? ''
  const isPersistedOnly = !file && !!persisted?.url

  useEffect(() => {
    setZoom(1)
    if (scrollRef.current) {
      scrollRef.current.scrollTop = 0
      scrollRef.current.scrollLeft = 0
    }
  }, [file, persisted?.url, generated?.url])

  useEffect(() => {
    const el = scrollRef.current
    if (!el || !displayUrl) return undefined

    const onWheel = (e) => {
      if (!e.ctrlKey && !e.metaKey) return
      e.preventDefault()
      setZoom((z) => {
        const delta = -e.deltaY * 0.002
        const next = Math.min(ZOOM_MAX, Math.max(ZOOM_MIN, z + delta))
        return Math.round(next / ZOOM_STEP) * ZOOM_STEP
      })
    }
    el.addEventListener('wheel', onWheel, { passive: false })
    return () => el.removeEventListener('wheel', onWheel)
  }, [displayUrl])

  if (!displayUrl) return null

  return (
    <Box sx={{ mt: 1 }}>
      <Typography variant="body2" sx={{ mb: 0.5, fontWeight: 500 }} noWrap title={displayName}>
        {displayName}
      </Typography>
      {isPersistedOnly && (
        <Typography variant="caption" color="success.main" sx={{ display: 'block', mb: 0.5 }}>
          已生成特征，预览保留；选择新图像可替换
        </Typography>
      )}
      <Typography variant="caption" color="text.secondary" sx={{ display: 'block', mb: 1 }}>
        图像预览（框内可拖动滚动；拖曳滑块或 Ctrl/⌘ + 滚轮缩放）
      </Typography>
      <Stack direction="row" spacing={1} alignItems="center" sx={{ mb: 1, flexWrap: 'wrap' }}>
        <IconButton
          size="small"
          aria-label="缩小"
          onClick={() => setZoom((z) => Math.max(ZOOM_MIN, z - 0.25))}
        >
          <ZoomOutIcon fontSize="small" />
        </IconButton>
        <Slider
          size="small"
          value={zoom}
          min={ZOOM_MIN}
          max={ZOOM_MAX}
          step={ZOOM_STEP}
          onChange={(_, v) => setZoom(v)}
          sx={{ flex: 1, minWidth: 120, maxWidth: 280 }}
        />
        <IconButton
          size="small"
          aria-label="放大"
          onClick={() => setZoom((z) => Math.min(ZOOM_MAX, z + 0.25))}
        >
          <ZoomInIcon fontSize="small" />
        </IconButton>
        <Typography variant="caption" color="text.secondary" sx={{ minWidth: 44 }}>
          {Math.round(zoom * 100)}%
        </Typography>
        <IconButton size="small" aria-label="重置为 100%" onClick={() => setZoom(1)} title="100%">
          <RestartAltIcon fontSize="small" />
        </IconButton>
      </Stack>

      <Box
        ref={scrollRef}
        sx={{
          border: 1,
          borderColor: 'divider',
          borderRadius: 1,
          overflow: 'auto',
          maxHeight: { xs: 280, sm: 360 },
          bgcolor: (t) => (t.palette.mode === 'dark' ? 'grey.900' : 'grey.100'),
          userSelect: 'none',
        }}
      >
        <Box sx={{ width: `${zoom * 100}%`, minWidth: '100%', p: 0.5, boxSizing: 'border-box' }}>
          <img
            src={displayUrl}
            alt="病理图像预览"
            draggable={false}
            style={{
              width: '100%',
              height: 'auto',
              display: 'block',
              verticalAlign: 'top',
            }}
          />
        </Box>
      </Box>
    </Box>
  )
}
