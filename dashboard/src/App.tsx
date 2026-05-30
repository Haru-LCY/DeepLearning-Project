import {
  Alert,
  Badge,
  Box,
  Button,
  Card,
  Container,
  FileUpload,
  Grid,
  GridItem,
  HStack,
  Icon,
  Link,
  Progress,
  RadioCard,
  SimpleGrid,
  Slider,
  Span,
  Spinner,
  Stack,
  Steps,
  Text,
  VStack,
} from "@chakra-ui/react"
import { useEffect, useRef, useState } from "react"
import {
  LuAudioLines,
  LuBadgeCheck,
  LuDownload,
  LuFileAudio,
  LuMicVocal,
  LuMoon,
  LuPiano,
  LuRefreshCw,
  LuRocket,
  LuSlidersHorizontal,
  LuSparkles,
  LuUpload,
  LuVolume2,
  LuCheck,
} from "react-icons/lu"

import { ColorModeButton } from "@/components/ui/color-mode"
import { toaster } from "@/components/ui/toaster"

const API_BASE = import.meta.env.VITE_API_BASE_URL ?? "http://localhost:8000"

type Role = {
  id: string
  name: string
  avatar?: string | null
  default_key: number
  ready: boolean
  loaded: boolean
  error?: string | null
}

type BackendConfig = {
  roles: Role[]
  constraints: {
    key: { min: number; max: number; step: number }
    vocals_volume: { min: number; max: number; step: number; default: number }
    piano_volume: { min: number; max: number; step: number; default: number }
  }
  stages: Array<{ id: number; name: string; label: string }>
}

type JobStatus = "queued" | "running" | "completed" | "failed" | "cancelled"

type Job = {
  job_id: string
  status: JobStatus
  stage: number | null
  stage_name: string | null
  progress: number
  message: string
  params: {
    role_id: string
    key: number
    vocals_volume: number
    piano_volume: number
    original_filename?: string | null
  }
  artifacts: {
    input_audio?: string | null
    vocals?: string | null
    piano?: string | null
    final?: string | null
  }
  error?: string | null
  updated_at?: string
}

const FALLBACK_CONFIG: BackendConfig = {
  roles: [
    { id: "amoris", name: "Amoris", default_key: 0, ready: false, loaded: false },
    { id: "anon", name: "Anon", default_key: 0, ready: false, loaded: false },
    { id: "doloris", name: "Doloris", default_key: 0, ready: false, loaded: false },
    { id: "mortis", name: "Mortis", default_key: 0, ready: false, loaded: false },
    { id: "soyo", name: "Soyo", default_key: 0, ready: false, loaded: false },
    { id: "taki", name: "Taki", default_key: 0, ready: false, loaded: false },
    { id: "tomorin", name: "Tomorin", default_key: 0, ready: false, loaded: false },
    { id: "oblivionis", name: "Oblivionis", default_key: 0, ready: false, loaded: false },
  ],
  constraints: {
    key: { min: -12, max: 12, step: 1 },
    vocals_volume: { min: 0, max: 2, step: 0.05, default: 1 },
    piano_volume: { min: 0, max: 2, step: 0.05, default: 1 },
  },
  stages: [
    { id: 1, name: "separate_vocals", label: "Separate vocals" },
    { id: 2, name: "voice_conversion", label: "Voice conversion" },
    { id: 3, name: "piano_cover", label: "Piano cover" },
    { id: 4, name: "merge", label: "Final mix" },
  ],
}

const STAGE_COPY: Record<string, string> = {
  separate_vocals: "人声分离",
  voice_conversion: "角色翻唱",
  piano_cover: "钢琴伴奏",
  merge: "最终混音",
  completed: "完成",
}

const STAGE_ICONS = [LuAudioLines, LuMicVocal, LuPiano, LuSlidersHorizontal]

function App() {
  const [config, setConfig] = useState<BackendConfig>(FALLBACK_CONFIG)
  const [selectedRole, setSelectedRole] = useState("amoris")
  const [audioFile, setAudioFile] = useState<File | null>(null)
  const [audioPreviewUrl, setAudioPreviewUrl] = useState<string | null>(null)
  const [keyShift, setKeyShift] = useState([0])
  const [vocalsVolume, setVocalsVolume] = useState([1])
  const [pianoVolume, setPianoVolume] = useState([1])
  const [job, setJob] = useState<Job | null>(null)
  const [submitting, setSubmitting] = useState(false)
  const [remixing, setRemixing] = useState(false)

  const selectedRoleDetails = config.roles.find((role) => role.id === selectedRole)
  const canSubmit = Boolean(audioFile && selectedRoleDetails?.ready && !submitting)
  const isRunning = job?.status === "queued" || job?.status === "running"
  const canRemix = job?.status === "completed" && Boolean(job.artifacts.final)
  const eventJobId = job?.job_id
  const eventJobStatus = job?.status
  const audioPreviewUrlRef = useRef<string | null>(null)

  useEffect(() => {
    const controller = new AbortController()

    async function loadConfig() {
      try {
        const response = await fetch(apiUrl("/api/config"), {
          signal: controller.signal,
        })
        if (!response.ok) {
          throw new Error(`Config request failed with ${response.status}`)
        }
        const nextConfig = (await response.json()) as BackendConfig
        setConfig(nextConfig)
        const firstReadyRole = nextConfig.roles.find((role) => role.ready)
        if (firstReadyRole) {
          setSelectedRole(firstReadyRole.id)
          setKeyShift([firstReadyRole.default_key])
        }
      } catch {
        if (!controller.signal.aborted) {
          toaster.create({
            title: "后端不可用",
            type: "warning",
          })
        }
      }
    }

    loadConfig()
    return () => controller.abort()
  }, [])

  useEffect(() => {
    return () => {
      if (audioPreviewUrlRef.current) {
        URL.revokeObjectURL(audioPreviewUrlRef.current)
      }
    }
  }, [])

  useEffect(() => {
    if (!eventJobId || eventJobStatus === "completed" || eventJobStatus === "failed") return

    const events = new EventSource(apiUrl(`/api/jobs/${eventJobId}/events`))
    const updateJob = (event: MessageEvent<string>) => {
      setJob(JSON.parse(event.data) as Job)
    }

    events.addEventListener("snapshot", updateJob)
    events.addEventListener("progress", updateJob)
    events.addEventListener("completed", updateJob)
    events.addEventListener("failed", updateJob)
    events.addEventListener("cancelled", updateJob)
    events.onerror = () => {
      events.close()
    }

    return () => events.close()
  }, [eventJobId, eventJobStatus])

  async function handleSubmit() {
    if (!audioFile) {
      toaster.create({
        title: "请选择音频文件",
        type: "warning",
      })
      return
    }

    try {
      setSubmitting(true)
      const formData = new FormData()
      formData.append("audio", audioFile)
      formData.append("role_id", selectedRole)
      formData.append("key", String(keyShift[0]))
      formData.append("vocals_volume", String(vocalsVolume[0]))
      formData.append("piano_volume", String(pianoVolume[0]))

      const response = await fetch(apiUrl("/api/jobs"), {
        method: "POST",
        body: formData,
      })

      if (!response.ok) {
        throw new Error(await readError(response))
      }

      const nextJob = (await response.json()) as Job
      setJob(nextJob)
      toaster.create({
        title: "任务已提交",
        type: "success",
      })
    } catch {
      toaster.create({
        title: "提交失败",
        type: "error",
      })
    } finally {
      setSubmitting(false)
    }
  }

  function handleAudioFileChange(file: File | null) {
    if (audioPreviewUrlRef.current) {
      URL.revokeObjectURL(audioPreviewUrlRef.current)
      audioPreviewUrlRef.current = null
    }

    setAudioFile(file)
    if (!file) {
      setAudioPreviewUrl(null)
      return
    }

    const nextPreviewUrl = URL.createObjectURL(file)
    audioPreviewUrlRef.current = nextPreviewUrl
    setAudioPreviewUrl(nextPreviewUrl)
  }

  async function handleRemix() {
    if (!job) return

    try {
      setRemixing(true)
      const response = await fetch(apiUrl(`/api/jobs/${job.job_id}/remix`), {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          vocals_volume: vocalsVolume[0],
          piano_volume: pianoVolume[0],
        }),
      })

      if (!response.ok) {
        throw new Error(await readError(response))
      }

      setJob((await response.json()) as Job)
      toaster.create({
        title: "混音已更新",
        type: "success",
      })
    } catch {
      toaster.create({
        title: "Remix 失败",
        type: "error",
      })
    } finally {
      setRemixing(false)
    }
  }

  return (
    <Box minH="100vh" bg="bg" color="fg" position="relative" overflow="hidden">
      <Box
        position="absolute"
        inset="0"
        bg="radial-gradient(circle at 15% 10%, rgba(129, 140, 248, 0.22), transparent 32%), radial-gradient(circle at 85% 0%, rgba(236, 72, 153, 0.16), transparent 28%), radial-gradient(circle at 50% 110%, rgba(45, 212, 191, 0.14), transparent 32%)"
        pointerEvents="none"
      />

      <Container maxW="7xl" py={{ base: 5, md: 8 }} position="relative">
        <Stack gap={{ base: 6, md: 8 }}>
          <HStack justify="flex-end">
            <ColorModeButton />
          </HStack>

          <Grid templateColumns={{ base: "1fr", xl: "1.18fr 0.82fr" }} gap="6">
            <GridItem>
              <Stack gap="6">
                <WorkflowCard job={job} stages={config.stages} />
                <UploadCard
                  audioFile={audioFile}
                  previewUrl={audioPreviewUrl}
                  onFileChange={handleAudioFileChange}
                  disabled={isRunning}
                />
                <RolePicker
                  roles={config.roles}
                  value={selectedRole}
                  onChange={(roleId) => {
                    const role = config.roles.find((item) => item.id === roleId)
                    setSelectedRole(roleId)
                    setKeyShift([role?.default_key ?? 0])
                  }}
                  disabled={isRunning}
                />
              </Stack>
            </GridItem>

            <GridItem>
              <Stack gap="6" position={{ xl: "sticky" }} top="6">
                <ControlPanel
                  constraints={config.constraints}
                  keyShift={keyShift}
                  vocalsVolume={vocalsVolume}
                  pianoVolume={pianoVolume}
                  onKeyShiftChange={setKeyShift}
                  onVocalsVolumeChange={setVocalsVolume}
                  onPianoVolumeChange={setPianoVolume}
                  canSubmit={canSubmit}
                  submitting={submitting}
                  remixing={remixing}
                  canRemix={canRemix}
                  onSubmit={handleSubmit}
                  onRemix={handleRemix}
                />
                <ResultCard job={job} />
              </Stack>
            </GridItem>
          </Grid>
        </Stack>
      </Container>
    </Box>
  )
}

type WorkflowCardProps = {
  job: Job | null
  stages: BackendConfig["stages"]
}

function WorkflowCard({ job, stages }: WorkflowCardProps) {
  const activeStep = Math.max((job?.stage ?? 1) - 1, 0)
  const progress = job?.progress ?? 0
  const running = job?.status === "running" || job?.status === "queued"

  return (
    <Card.Root>
      <Card.Header pb="3">
        <HStack justify="space-between" align="flex-start">
          <Stack gap="1">
            <Card.Title>四阶段处理进度</Card.Title>
          </Stack>
          {running && <Spinner size="sm" />}
        </HStack>
      </Card.Header>
      <Card.Body gap="5">
        <Steps.Root step={activeStep} count={stages.length} colorPalette="purple" size="sm">
          <Steps.List>
            {stages.map((stage, index) => {
              const StageIcon = STAGE_ICONS[index] ?? LuSparkles
              return (
                <Steps.Item key={stage.name} index={index} title={stage.label}>
                  <Steps.Indicator>
                    <Steps.Status complete={<LuCheck />} incomplete={<StageIcon />} />
                  </Steps.Indicator>
                  <Steps.Title display={{ base: "none", md: "block" }}>
                    {STAGE_COPY[stage.name] ?? stage.label}
                  </Steps.Title>
                  <Steps.Separator />
                </Steps.Item>
              )
            })}
          </Steps.List>
        </Steps.Root>

        <Progress.Root value={progress} colorPalette="purple" striped={running} animated={running}>
          <HStack justify="space-between" mb="2">
            <Progress.Label color="fg.muted">
              {job?.stage_name ? STAGE_COPY[job.stage_name] ?? job.stage_name : "等待提交"}
            </Progress.Label>
            <Progress.ValueText>{progress}%</Progress.ValueText>
          </HStack>
          <Progress.Track>
            <Progress.Range />
          </Progress.Track>
        </Progress.Root>

        {job?.status === "failed" && (
          <Alert.Root status="error" variant="surface">
            <Alert.Indicator />
            <Alert.Content>
              <Alert.Title>{job.error ?? job.message}</Alert.Title>
            </Alert.Content>
          </Alert.Root>
        )}
      </Card.Body>
    </Card.Root>
  )
}

type UploadCardProps = {
  audioFile: File | null
  previewUrl: string | null
  onFileChange: (file: File | null) => void
  disabled: boolean
}

function UploadCard({ audioFile, previewUrl, onFileChange, disabled }: UploadCardProps) {
  return (
    <Card.Root>
      <Card.Header pb="3">
        <Card.Title>上传输入音频</Card.Title>
      </Card.Header>
      <Card.Body>
        <FileUpload.Root
          accept={["audio/*"]}
          maxFiles={1}
          disabled={disabled}
          onFileChange={(details) => onFileChange(details.acceptedFiles[0] ?? null)}
          alignItems="stretch"
        >
          <FileUpload.HiddenInput />
          <FileUpload.Dropzone minH="180px" borderStyle="dashed" bg="bg.subtle/60">
            <Icon fontSize="3xl" color="purple.fg">
              <LuUpload />
            </Icon>
            <FileUpload.DropzoneContent>
              <Text fontWeight="semibold">拖拽音频到这里，或点击选择文件</Text>
            </FileUpload.DropzoneContent>
          </FileUpload.Dropzone>
          <FileUpload.List showSize clearable />
        </FileUpload.Root>

        {audioFile && (
          <Stack mt="4" p="3" rounded="lg" bg="bg.subtle" gap="3">
            <HStack gap="3">
              <Icon color="purple.fg">
                <LuFileAudio />
              </Icon>
              <Stack gap="0" minW="0">
                <Text fontWeight="medium" truncate>
                  {audioFile.name}
                </Text>
                <Text color="fg.muted" textStyle="sm">
                  {formatFileSize(audioFile.size)} ready
                </Text>
              </Stack>
            </HStack>
            {previewUrl && (
              <audio controls src={previewUrl} style={{ width: "100%" }} />
            )}
          </Stack>
        )}
      </Card.Body>
    </Card.Root>
  )
}

type RolePickerProps = {
  roles: Role[]
  value: string
  onChange: (value: string) => void
  disabled: boolean
}

function RolePicker({ roles, value, onChange, disabled }: RolePickerProps) {
  return (
    <Card.Root>
      <Card.Header pb="3">
        <Card.Title>选择翻唱角色</Card.Title>
      </Card.Header>
      <Card.Body>
        <RadioCard.Root
          value={value}
          onValueChange={(details) => {
            if (details.value) onChange(details.value)
          }}
          colorPalette="purple"
          disabled={disabled}
        >
          <SimpleGrid columns={{ base: 1, sm: 2, lg: 4 }} gap="3">
            {roles.map((role) => (
              <RadioCard.Item key={role.id} value={role.id} disabled={!role.ready}>
                <RadioCard.ItemHiddenInput />
                <RadioCard.ItemControl minH="76px" alignItems="center">
                  <RadioCard.ItemContent>
                    <RadioCard.ItemText>{role.name}</RadioCard.ItemText>
                  </RadioCard.ItemContent>
                  <RadioCard.ItemIndicator />
                </RadioCard.ItemControl>
                {!role.ready && role.error && (
                  <RadioCard.ItemAddon color="orange.fg" textStyle="xs" lineClamp={2}>
                    {role.error}
                  </RadioCard.ItemAddon>
                )}
              </RadioCard.Item>
            ))}
          </SimpleGrid>
        </RadioCard.Root>
      </Card.Body>
    </Card.Root>
  )
}

type ControlPanelProps = {
  constraints: BackendConfig["constraints"]
  keyShift: number[]
  vocalsVolume: number[]
  pianoVolume: number[]
  onKeyShiftChange: (value: number[]) => void
  onVocalsVolumeChange: (value: number[]) => void
  onPianoVolumeChange: (value: number[]) => void
  canSubmit: boolean
  submitting: boolean
  remixing: boolean
  canRemix: boolean
  onSubmit: () => void
  onRemix: () => void
}

function ControlPanel(props: ControlPanelProps) {
  const {
    constraints,
    keyShift,
    vocalsVolume,
    pianoVolume,
    onKeyShiftChange,
    onVocalsVolumeChange,
    onPianoVolumeChange,
    canSubmit,
    submitting,
    remixing,
    canRemix,
    onSubmit,
    onRemix,
  } = props

  return (
    <Card.Root shadow="xl">
      <Card.Header pb="3">
        <HStack justify="space-between">
          <Stack gap="1">
            <Card.Title>参数控制</Card.Title>
          </Stack>
          <Icon color="purple.fg" fontSize="2xl">
            <LuSlidersHorizontal />
          </Icon>
        </HStack>
      </Card.Header>
      <Card.Body gap="6">
        <Stack gap="5">
          <ParameterSlider
            label="升降 Key"
            value={keyShift}
            min={constraints.key.min}
            max={constraints.key.max}
            step={constraints.key.step}
            marks={[
              { value: constraints.key.min, label: String(constraints.key.min) },
              { value: 0, label: "0" },
              { value: constraints.key.max, label: `+${constraints.key.max}` },
            ]}
            valueText={`${formatSigned(keyShift[0])} semitones`}
            colorPalette="purple"
            origin="center"
            onChange={onKeyShiftChange}
          />

          <VolumeSlider
            label="翻唱人声音量"
            value={vocalsVolume}
            min={constraints.vocals_volume.min}
            max={constraints.vocals_volume.max}
            step={constraints.vocals_volume.step}
            onChange={onVocalsVolumeChange}
            icon={<LuVolume2 />}
          />
          <VolumeSlider
            label="Piano cover 音量"
            value={pianoVolume}
            min={constraints.piano_volume.min}
            max={constraints.piano_volume.max}
            step={constraints.piano_volume.step}
            onChange={onPianoVolumeChange}
            icon={<LuPiano />}
          />
        </Stack>
      </Card.Body>
      <Card.Footer flexDir="column" alignItems="stretch" gap="3">
        <Button
          size="lg"
          colorPalette="purple"
          disabled={!canSubmit}
          loading={submitting}
          onClick={onSubmit}
        >
          <LuRocket />
          提交完整任务
        </Button>
        <Button
          variant="outline"
          disabled={!canRemix}
          loading={remixing}
          onClick={onRemix}
        >
          <LuRefreshCw />
          仅重新混音第四阶段
        </Button>
      </Card.Footer>
    </Card.Root>
  )
}

type VolumeSliderProps = {
  label: string
  value: number[]
  min: number
  max: number
  step: number
  icon: React.ReactNode
  onChange: (value: number[]) => void
}

function VolumeSlider({ label, value, min, max, step, icon, onChange }: VolumeSliderProps) {
  return (
    <ParameterSlider
      label={label}
      value={value}
      min={min}
      max={max}
      step={step}
      marks={[
        { value: min, label: min.toFixed(0) },
        { value: 1, label: "1x" },
        { value: max, label: `${max.toFixed(0)}x` },
      ]}
      valueText={`${value[0].toFixed(2)}x`}
      colorPalette="cyan"
      icon={icon}
      onChange={onChange}
    />
  )
}

type ParameterSliderProps = {
  label: string
  value: number[]
  min: number
  max: number
  step: number
  marks: Array<{ value: number; label: string }>
  valueText: string
  colorPalette: "purple" | "cyan"
  icon?: React.ReactNode
  origin?: "center" | "start" | "end"
  onChange: (value: number[]) => void
}

function ParameterSlider({
  label,
  value,
  min,
  max,
  step,
  marks,
  valueText,
  colorPalette,
  icon,
  origin = "start",
  onChange,
}: ParameterSliderProps) {
  return (
    <Slider.Root
      value={value}
      min={min}
      max={max}
      step={step}
      origin={origin}
      colorPalette={colorPalette}
      thumbAlignment="center"
      getAriaValueText={(details) => String(details.value)}
      onValueChange={(details) => onChange(details.value)}
    >
      <HStack justify="space-between" mb="3">
        <Slider.Label>
          <HStack gap="2">
            {icon && <Icon color="fg.muted">{icon}</Icon>}
            <Span>{label}</Span>
          </HStack>
        </Slider.Label>
        <Badge colorPalette={colorPalette} variant="surface">
          {valueText}
        </Badge>
      </HStack>
      <Slider.Control data-has-mark-label>
        <Slider.Track>
          <Slider.Range />
        </Slider.Track>
        <Slider.Thumb index={0}>
          <Slider.HiddenInput />
        </Slider.Thumb>
        <Slider.MarkerGroup>
          {marks.map((mark) => (
            <Slider.Marker key={`${label}-${mark.value}`} value={mark.value}>
              <Slider.MarkerIndicator />
              <Slider.MarkerLabel color="fg.muted" textStyle="xs">
                {mark.label}
              </Slider.MarkerLabel>
            </Slider.Marker>
          ))}
        </Slider.MarkerGroup>
      </Slider.Control>
    </Slider.Root>
  )
}

type ResultCardProps = {
  job: Job | null
}

function ResultCard({ job }: ResultCardProps) {
  const artifactVersion = job?.updated_at ?? `${job?.progress ?? 0}`

  return (
    <Card.Root>
      <Card.Header pb="3">
        <HStack justify="space-between">
          <Stack gap="1">
            <Card.Title>结果预览</Card.Title>
          </Stack>
          {job?.status === "completed" && (
            <Badge colorPalette="green" variant="surface">
              <LuBadgeCheck /> Completed
            </Badge>
          )}
        </HStack>
      </Card.Header>
      <Card.Body gap="4">
        {!job && (
          <VStack py="8" gap="3" color="fg.muted">
            <Icon fontSize="4xl">
              <LuMoon />
            </Icon>
          </VStack>
        )}

        {job && (
          <Stack gap="4">
            <HStack justify="space-between" align="center">
              <Stack gap="0">
                <Text fontWeight="semibold">Job {job.job_id.slice(0, 8)}</Text>
                <Text color="fg.muted" textStyle="sm">
                  {job.params.original_filename ?? "input audio"}
                </Text>
              </Stack>
              <Badge colorPalette={job.status === "failed" ? "red" : job.status === "completed" ? "green" : "purple"}>
                {job.status}
              </Badge>
            </HStack>

            {job.artifacts.final && (
              <Box p="4" rounded="xl" bg="bg.subtle">
                <Text mb="2" fontWeight="medium">
                  Final mix
                </Text>
                <audio
                  key={artifactVersion}
                  controls
                  src={artifactUrl(job.artifacts.final, artifactVersion)}
                  style={{ width: "100%" }}
                />
              </Box>
            )}

            <SimpleGrid columns={{ base: 1, md: 3, xl: 1 }} gap="3">
              <DownloadButton label="翻唱人声" href={job.artifacts.vocals} version={artifactVersion} />
              <DownloadButton label="钢琴伴奏" href={job.artifacts.piano} version={artifactVersion} />
              <DownloadButton label="最终混音" href={job.artifacts.final} version={artifactVersion} />
            </SimpleGrid>
          </Stack>
        )}
      </Card.Body>
    </Card.Root>
  )
}

function DownloadButton({
  label,
  href,
  version,
}: {
  label: string
  href?: string | null
  version: string
}) {
  return (
    <Button asChild variant="outline" disabled={!href} justifyContent="space-between">
      <Link href={href ? artifactUrl(href, version) : undefined} download>
        {label}
        <LuDownload />
      </Link>
    </Button>
  )
}

function apiUrl(path: string) {
  if (path.startsWith("http://") || path.startsWith("https://")) return path
  return `${API_BASE}${path}`
}

function artifactUrl(path: string, version: string) {
  const url = apiUrl(path)
  const separator = url.includes("?") ? "&" : "?"
  return `${url}${separator}v=${encodeURIComponent(version)}`
}

async function readError(response: Response) {
  try {
    const body = (await response.json()) as { detail?: string }
    return body.detail ?? response.statusText
  } catch {
    return response.statusText
  }
}

function formatFileSize(size: number) {
  if (size < 1024 * 1024) return `${(size / 1024).toFixed(1)} KB`
  return `${(size / 1024 / 1024).toFixed(1)} MB`
}

function formatSigned(value: number) {
  return value > 0 ? `+${value}` : String(value)
}

export default App
