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
  Image,
  Input,
  Link,
  Progress,
  SimpleGrid,
  Slider,
  Span,
  Spinner,
  Stack,
  Steps,
  Text,
  VStack,
} from "@chakra-ui/react"
import { useEffect, useMemo, useRef, useState } from "react"
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
  LuServer,
  LuSlidersHorizontal,
  LuSparkles,
  LuUpload,
  LuVolume2,
  LuCheck,
} from "react-icons/lu"

import amorisImage from "../character/amoris.png"
import anonImage from "../character/anon.png"
import dolorisImage from "../character/doloris.png"
import mortisImage from "../character/mortis.png"
import oblivionisImage from "../character/oblivionis.png"
import soyoImage from "../character/soyo.png"
import takiImage from "../character/taki.png"
import tomorinImage from "../character/tomorin.png"

import { ColorModeButton, useColorMode } from "@/components/ui/color-mode"
import { toaster } from "@/components/ui/toaster"

const DEFAULT_API_BASE = import.meta.env.VITE_API_BASE_URL ?? "http://localhost:8000"

function loadApiBase(): string {
  return localStorage.getItem("api_base") || DEFAULT_API_BASE
}

let _apiBase = loadApiBase()

type Role = {
  id: string
  name: string
  avatar?: string | null
  default_pre_pitch_shift: number
  ready: boolean
  loaded: boolean
  error?: string | null
}

type BackendConfig = {
  roles: Role[]
  constraints: {
    pre_pitch_shift: { min: number; max: number; step: number }
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
    pre_pitch_shift: number
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
    { id: "amoris", name: "Amoris", default_pre_pitch_shift: 0, ready: false, loaded: false },
    { id: "anon", name: "Anon", default_pre_pitch_shift: 0, ready: false, loaded: false },
    { id: "doloris", name: "Doloris", default_pre_pitch_shift: 0, ready: false, loaded: false },
    { id: "mortis", name: "Mortis", default_pre_pitch_shift: 0, ready: false, loaded: false },
    { id: "soyo", name: "Soyo", default_pre_pitch_shift: 0, ready: false, loaded: false },
    { id: "taki", name: "Taki", default_pre_pitch_shift: 0, ready: false, loaded: false },
    { id: "tomorin", name: "Tomorin", default_pre_pitch_shift: 0, ready: false, loaded: false },
    { id: "oblivionis", name: "Oblivionis", default_pre_pitch_shift: 0, ready: false, loaded: false },
  ],
  constraints: {
    pre_pitch_shift: { min: -12, max: 12, step: 1 },
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

const LIGHT_ROLE_IDS = ["anon", "soyo", "taki", "tomorin"] as const
const DARK_ROLE_IDS = ["amoris", "doloris", "oblivionis", "mortis"] as const

type CharacterRoleId = (typeof LIGHT_ROLE_IDS)[number] | (typeof DARK_ROLE_IDS)[number]
type AccentPalette = "blue" | "red"

const CHARACTER_ROLE_IMAGES: Record<CharacterRoleId, string> = {
  amoris: amorisImage,
  anon: anonImage,
  doloris: dolorisImage,
  mortis: mortisImage,
  oblivionis: oblivionisImage,
  soyo: soyoImage,
  taki: takiImage,
  tomorin: tomorinImage,
}

const CHARACTER_IMAGE_POSITIONS: Partial<Record<CharacterRoleId, string>> = {
  mortis: "center 18%",
  oblivionis: "center 18%",
  tomorin: "center top",
}

function panelCardBg(accentPalette: AccentPalette) {
  return accentPalette === "red" ? "rgba(42, 8, 12, 0.92)" : "rgba(255, 255, 255, 0.78)"
}

function panelBorderColor(accentPalette: AccentPalette) {
  return accentPalette === "red" ? "rgba(248, 113, 113, 0.30)" : "rgba(147, 197, 253, 0.42)"
}

function panelBodyBg(accentPalette: AccentPalette) {
  return accentPalette === "red"
    ? "radial-gradient(circle at 18% 0%, rgba(92, 18, 24, 0.46), transparent 34%), radial-gradient(circle at 84% 12%, rgba(52, 9, 12, 0.66), transparent 30%), linear-gradient(180deg, rgba(45, 9, 13, 0.94), rgba(20, 7, 10, 0.98))"
    : "radial-gradient(circle at 20% 0%, rgba(191, 219, 254, 0.45), transparent 34%), radial-gradient(circle at 82% 8%, rgba(219, 234, 254, 0.34), transparent 28%), linear-gradient(180deg, rgba(248, 250, 252, 0.62), rgba(240, 249, 255, 0.48))"
}

function panelInnerBg(accentPalette: AccentPalette) {
  return accentPalette === "red" ? "rgba(37, 7, 10, 0.56)" : "rgba(240, 249, 255, 0.66)"
}

function controlSolidBg(accentPalette: AccentPalette) {
  return accentPalette === "red" ? "#7f1d1d" : undefined
}

function controlSolidHoverBg(accentPalette: AccentPalette) {
  return accentPalette === "red" ? "#991b1b" : undefined
}

function controlSoftBg(accentPalette: AccentPalette) {
  return accentPalette === "red" ? "#5f151b" : undefined
}

function controlAccentColor(accentPalette: AccentPalette) {
  return accentPalette === "red" ? "#b45353" : `${accentPalette}.fg`
}

function App() {
  const { colorMode } = useColorMode()
  const isDarkMode = colorMode === "dark"
  const accentPalette: AccentPalette = isDarkMode ? "red" : "blue"
  const pageGradient = isDarkMode
    ? "radial-gradient(circle at 18% 0%, rgba(127, 29, 29, 0.26), transparent 34%), radial-gradient(circle at 84% 8%, rgba(69, 10, 10, 0.46), transparent 30%), linear-gradient(180deg, rgba(17, 7, 10, 0.98), rgba(36, 7, 12, 0.94))"
    : "radial-gradient(circle at 15% 8%, rgba(186, 230, 253, 0.42), transparent 34%), radial-gradient(circle at 88% 0%, rgba(219, 234, 254, 0.48), transparent 30%), radial-gradient(circle at 50% 110%, rgba(125, 211, 252, 0.22), transparent 34%)"

  const [apiBase, setApiBase] = useState(loadApiBase)

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
          setKeyShift([firstReadyRole.default_pre_pitch_shift])
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
  }, [apiBase])

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
      formData.append("pre_pitch_shift", String(keyShift[0]))
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

  function handleSaveApiBase() {
    const trimmed = apiBase.trim() || DEFAULT_API_BASE
    localStorage.setItem("api_base", trimmed)
    _apiBase = trimmed
    setApiBase(trimmed)
    toaster.create({
      title: "后端地址已保存",
      type: "success",
    })
  }

  return (
    <Box minH="100vh" bg={isDarkMode ? "#100709" : "bg"} color="fg" position="relative" overflow="hidden">
      <Box
        position="absolute"
        inset="0"
        bg={pageGradient}
        pointerEvents="none"
      />

      <Container maxW="7xl" pt={{ base: 3, md: 4 }} pb={{ base: 5, md: 8 }} position="relative">
        <Stack gap={{ base: 4, md: 5 }}>
          <Grid templateColumns={{ base: "1fr", xl: "1.18fr 0.82fr" }} gap="6">
            <GridItem>
              <Stack gap="6">
                <WorkflowCard job={job} stages={config.stages} accentPalette={accentPalette} />
                <RolePicker
                  roles={config.roles}
                  value={selectedRole}
                  onChange={(roleId) => {
                    const role = config.roles.find((item) => item.id === roleId)
                    setSelectedRole(roleId)
                    setKeyShift([role?.default_pre_pitch_shift ?? 0])
                  }}
                  disabled={isRunning}
                />
              </Stack>
            </GridItem>

            <GridItem>
              <Stack gap="4" position={{ xl: "sticky" }} top="6">
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
                  accentPalette={accentPalette}
                />
                <UploadCard
                  audioFile={audioFile}
                  previewUrl={audioPreviewUrl}
                  onFileChange={handleAudioFileChange}
                  disabled={isRunning}
                  accentPalette={accentPalette}
                />
                <ResultCard job={job} accentPalette={accentPalette} />
              </Stack>
            </GridItem>
          </Grid>

          <HStack justify="space-between" gap="3" flexWrap="wrap">
            <HStack gap="2">
              <Icon color="fg.muted" size="sm">
                <LuServer />
              </Icon>
              <Input
                size="xs"
                width="220px"
                placeholder={DEFAULT_API_BASE}
                value={apiBase}
                onChange={(e) => setApiBase(e.target.value)}
                onBlur={handleSaveApiBase}
                onKeyDown={(e) => {
                  if (e.key === "Enter") handleSaveApiBase()
                }}
              />
            </HStack>
            <ColorModeButton />
          </HStack>
        </Stack>
      </Container>
    </Box>
  )
}

type WorkflowCardProps = {
  job: Job | null
  stages: BackendConfig["stages"]
  accentPalette: AccentPalette
}

function WorkflowCard({ job, stages, accentPalette }: WorkflowCardProps) {
  const activeStep = Math.max((job?.stage ?? 1) - 1, 0)
  const progress = job?.progress ?? 0
  const running = job?.status === "running" || job?.status === "queued"

  return (
    <Card.Root
      overflow="hidden"
      bg={panelCardBg(accentPalette)}
      borderWidth="1px"
      borderColor={panelBorderColor(accentPalette)}
      shadow="sm"
    >
      <Card.Header pb="3">
        <HStack justify="space-between" align="flex-start">
          <Stack gap="1">
            <Card.Title>四阶段处理进度</Card.Title>
          </Stack>
          {running && <Spinner size="sm" color={controlAccentColor(accentPalette)} />}
        </HStack>
      </Card.Header>
      <Card.Body gap="5" bg={panelBodyBg(accentPalette)}>
        <Steps.Root step={activeStep} count={stages.length} colorPalette={accentPalette} size="sm">
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

        <Progress.Root value={progress} colorPalette={accentPalette} striped={running} animated={running}>
          <HStack justify="space-between" mb="2">
            <Progress.Label color="fg.muted">
              {job?.stage_name ? STAGE_COPY[job.stage_name] ?? job.stage_name : "等待提交"}
            </Progress.Label>
            <Progress.ValueText>{progress}%</Progress.ValueText>
          </HStack>
          <Progress.Track>
            <Progress.Range bg={controlSolidBg(accentPalette)} />
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
  accentPalette: AccentPalette
}

function UploadCard({ audioFile, previewUrl, onFileChange, disabled, accentPalette }: UploadCardProps) {
  return (
    <Card.Root
      overflow="hidden"
      bg={panelCardBg(accentPalette)}
      borderWidth="1px"
      borderColor={panelBorderColor(accentPalette)}
      shadow="sm"
    >
      <Card.Header py="3" pb="2">
        <Card.Title>上传输入音频</Card.Title>
      </Card.Header>
      <Card.Body pt="2" bg={panelBodyBg(accentPalette)}>
        <FileUpload.Root
          accept={["audio/*"]}
          maxFiles={1}
          disabled={disabled}
          onFileChange={(details) => onFileChange(details.acceptedFiles[0] ?? null)}
          alignItems="stretch"
        >
          <FileUpload.HiddenInput />
          <FileUpload.Dropzone minH="92px" py="2" borderStyle="dashed" borderColor={panelBorderColor(accentPalette)} bg={panelInnerBg(accentPalette)}>
            <Icon fontSize="xl" color={controlAccentColor(accentPalette)}>
              <LuUpload />
            </Icon>
            <FileUpload.DropzoneContent>
              <Text fontWeight="semibold" textStyle="sm">拖拽音频到这里，或点击选择文件</Text>
            </FileUpload.DropzoneContent>
          </FileUpload.Dropzone>
          <FileUpload.List showSize clearable />
        </FileUpload.Root>

        {audioFile && (
          <Stack mt="2" p="2" rounded="lg" bg={panelInnerBg(accentPalette)} gap="1">
            <HStack gap="3">
              <Icon color={controlAccentColor(accentPalette)}>
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
  const { colorMode } = useColorMode()
  const isDarkMode = colorMode === "dark"
  const visibleRoleIds = isDarkMode ? DARK_ROLE_IDS : LIGHT_ROLE_IDS
  const visibleRoles = useMemo(
    () =>
      visibleRoleIds.map((roleId) => roles.find((role) => role.id === roleId)).filter(
        (role): role is Role => Boolean(role),
      ),
    [roles, visibleRoleIds],
  )

  useEffect(() => {
    if (disabled || visibleRoles.length === 0) return
    if (visibleRoles.some((role) => role.id === value)) return

    const nextRole = visibleRoles.find((role) => role.ready) ?? visibleRoles[0]
    onChange(nextRole.id)
  }, [disabled, onChange, value, visibleRoles])

  return (
    <Card.Root
      overflow="hidden"
      bg={panelCardBg(isDarkMode ? "red" : "blue")}
      borderWidth="1px"
      borderColor={panelBorderColor(isDarkMode ? "red" : "blue")}
      shadow="sm"
    >
      <Card.Header pb="3">
        <Card.Title>选择翻唱角色</Card.Title>
      </Card.Header>
      <Card.Body bg={panelBodyBg(isDarkMode ? "red" : "blue")}>
        <SimpleGrid columns={{ base: 1, md: 2, xl: 4 }} gap={{ base: 3, md: 4 }}>
          {visibleRoles.map((role) => {
            const roleId = role.id as CharacterRoleId
            const selected = role.id === value
            const roleImage = CHARACTER_ROLE_IMAGES[roleId]
            const cardDisabled = disabled || !role.ready

            return (
              <Button
                key={role.id}
                type="button"
                variant="plain"
                disabled={cardDisabled}
                aria-pressed={selected}
                onClick={() => {
                  if (!cardDisabled) onChange(role.id)
                }}
                position="relative"
                display="block"
                h="auto"
                minH={{ base: "360px", md: "390px", xl: "420px" }}
                minW="0"
                p="0"
                overflow="hidden"
                rounded="2xl"
                borderWidth={selected ? "3px" : "2px"}
                borderColor={
                  selected
                    ? isDarkMode ? "#7f1d1d" : "blue.500"
                    : isDarkMode ? "rgba(248, 113, 113, 0.30)" : "rgba(147, 197, 253, 0.48)"
                }
                bg={
                  isDarkMode
                    ? "linear-gradient(145deg, #3f0b12 0%, #68131b 52%, #1f0509 100%)"
                    : "linear-gradient(145deg, #d9eef7 0%, #eef7f9 50%, #e9eefb 100%)"
                }
                boxShadow={
                  selected
                    ? isDarkMode ? "0 18px 42px rgba(248, 113, 113, 0.24)" : "0 18px 40px rgba(37, 99, 235, 0.24)"
                    : isDarkMode ? "0 12px 28px rgba(0, 0, 0, 0.28)" : "0 10px 24px rgba(15, 23, 42, 0.08)"
                }
                cursor={cardDisabled ? "not-allowed" : "pointer"}
                opacity={cardDisabled ? 0.58 : 1}
                transition="border-color 0.18s ease, box-shadow 0.18s ease, transform 0.18s ease"
                _hover={
                  cardDisabled
                    ? undefined
                    : {
                        transform: "translateY(-2px)",
                        borderColor: selected ? isDarkMode ? "#7f1d1d" : "blue.500" : isDarkMode ? "#b45353" : "blue.300",
                      }
                }
                _focusVisible={{
                  outline: "3px solid",
                  outlineColor: isDarkMode ? "#b45353" : "blue.300",
                  outlineOffset: "3px",
                }}
              >
                <Image
                  src={roleImage}
                  alt={role.name}
                  position="absolute"
                  inset="0"
                  w="100%"
                  h="100%"
                  objectFit="cover"
                  objectPosition={CHARACTER_IMAGE_POSITIONS[roleId] ?? "center 12%"}
                  pointerEvents="none"
                />
                <Box
                  position="absolute"
                  inset="0"
                  bg={
                    isDarkMode
                      ? "linear-gradient(180deg, rgba(24,5,8,0) 52%, rgba(24,5,8,0.18) 66%, rgba(24,5,8,0.58) 100%)"
                      : "linear-gradient(180deg, rgba(255,255,255,0) 58%, rgba(255,255,255,0.14) 68%, rgba(15,23,42,0.10) 100%)"
                  }
                  pointerEvents="none"
                />
                <Stack
                  position="absolute"
                  left="3"
                  right="3"
                  bottom="3"
                  minH="86px"
                  justify="center"
                  gap="1"
                  px="4"
                  py="3"
                  rounded="xl"
                  bg={isDarkMode ? "rgba(37, 7, 10, 0.82)" : "rgba(255, 255, 255, 0.84)"}
                  color={isDarkMode ? "red.50" : "gray.800"}
                  boxShadow={isDarkMode ? "0 10px 30px rgba(0, 0, 0, 0.34)" : "0 10px 28px rgba(15, 23, 42, 0.14)"}
                  backdropFilter="blur(8px)"
                >
                  <Text fontWeight="bold" fontSize={{ base: "xl", md: "2xl" }} color={isDarkMode ? "red.100" : "blue.600"}>
                    {role.name}
                  </Text>
                  {!role.ready && (
                    <Text color={isDarkMode ? "orange.200" : "orange.700"} textStyle="xs" lineClamp={2}>
                      {role.error ?? "角色暂不可用"}
                    </Text>
                  )}
                </Stack>
              </Button>
            )
          })}
        </SimpleGrid>
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
  accentPalette: AccentPalette
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
    accentPalette,
  } = props

  return (
    <Card.Root
      overflow="hidden"
      bg={panelCardBg(accentPalette)}
      borderWidth="1px"
      borderColor={panelBorderColor(accentPalette)}
      shadow="sm"
    >
      <Card.Header py="3" pb="2">
        <HStack justify="space-between">
          <Stack gap="1">
            <Card.Title>参数控制</Card.Title>
          </Stack>
          <Icon color={controlAccentColor(accentPalette)} fontSize="xl">
            <LuSlidersHorizontal />
          </Icon>
        </HStack>
      </Card.Header>
      <Card.Body pt="2" pb="3" gap="4" bg={panelBodyBg(accentPalette)}>
        <Stack gap="4">
          <ParameterSlider
            label="升降 Key"
            value={keyShift}
            min={constraints.pre_pitch_shift.min}
            max={constraints.pre_pitch_shift.max}
            step={constraints.pre_pitch_shift.step}
            marks={[
              { value: constraints.pre_pitch_shift.min, label: String(constraints.pre_pitch_shift.min) },
              { value: 0, label: "0" },
              { value: constraints.pre_pitch_shift.max, label: `+${constraints.pre_pitch_shift.max}` },
            ]}
            valueText={`${formatSigned(keyShift[0])} semitones`}
            colorPalette={accentPalette}
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
            accentPalette={accentPalette}
          />
          <VolumeSlider
            label="Piano cover 音量"
            value={pianoVolume}
            min={constraints.piano_volume.min}
            max={constraints.piano_volume.max}
            step={constraints.piano_volume.step}
            onChange={onPianoVolumeChange}
            icon={<LuPiano />}
            accentPalette={accentPalette}
          />
        </Stack>
      </Card.Body>
      <Card.Footer pt="2" flexDir="column" alignItems="stretch" gap="2" bg={panelBodyBg(accentPalette)}>
        <Button
          size="md"
          colorPalette={accentPalette}
          bg={controlSolidBg(accentPalette)}
          _hover={{ bg: controlSolidHoverBg(accentPalette) }}
          disabled={!canSubmit}
          loading={submitting}
          onClick={onSubmit}
        >
          <LuRocket />
          提交完整任务
        </Button>
        <Button
          variant="outline"
          colorPalette={accentPalette}
          borderColor={controlSoftBg(accentPalette)}
          color={controlAccentColor(accentPalette)}
          _hover={{ bg: controlSoftBg(accentPalette) }}
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
  accentPalette: AccentPalette
  onChange: (value: number[]) => void
}

function VolumeSlider({ label, value, min, max, step, icon, accentPalette, onChange }: VolumeSliderProps) {
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
      colorPalette={accentPalette}
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
  colorPalette: AccentPalette
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
      <HStack justify="space-between" mb="2">
        <Slider.Label>
          <HStack gap="2">
            {icon && <Icon color="fg.muted">{icon}</Icon>}
            <Span>{label}</Span>
          </HStack>
        </Slider.Label>
        <Badge colorPalette={colorPalette} variant="surface" bg={controlSoftBg(colorPalette)} color={controlAccentColor(colorPalette)}>
          {valueText}
        </Badge>
      </HStack>
      <Slider.Control data-has-mark-label>
        <Slider.Track>
          <Slider.Range bg={controlSolidBg(colorPalette)} />
        </Slider.Track>
        <Slider.Thumb
          index={0}
          bg={colorPalette === "red" ? "#2d070b" : undefined}
          borderColor={controlSolidBg(colorPalette)}
          boxShadow={colorPalette === "red" ? "0 0 0 3px rgba(127, 29, 29, 0.24)" : undefined}
        >
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
  accentPalette: AccentPalette
}

function ResultCard({ job, accentPalette }: ResultCardProps) {
  const artifactVersion = job?.updated_at ?? `${job?.progress ?? 0}`

  return (
    <Card.Root
      overflow="hidden"
      bg={panelCardBg(accentPalette)}
      borderWidth="1px"
      borderColor={panelBorderColor(accentPalette)}
      shadow="sm"
    >
      <Card.Header py="3" pb="2">
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
      <Card.Body pt="2" gap="3" bg={panelBodyBg(accentPalette)}>
        {!job && (
          <VStack py="3" gap="2" color="fg.muted">
            <Icon fontSize="2xl" color={controlAccentColor(accentPalette)}>
              <LuMoon />
            </Icon>
          </VStack>
        )}

        {job && (
          <Stack gap="3">
            <HStack justify="space-between" align="center">
              <Stack gap="0">
                <Text fontWeight="semibold">Job {job.job_id.slice(0, 8)}</Text>
                <Text color="fg.muted" textStyle="sm">
                  {job.params.original_filename ?? "input audio"}
                </Text>
              </Stack>
              <Badge colorPalette={job.status === "failed" ? "red" : job.status === "completed" ? "green" : accentPalette}
                bg={job.status === "running" || job.status === "queued" ? controlSoftBg(accentPalette) : undefined}
                color={job.status === "running" || job.status === "queued" ? controlAccentColor(accentPalette) : undefined}>
                {job.status}
              </Badge>
            </HStack>

            {job.artifacts.final && (
              <Box p="3" rounded="xl" bg={panelInnerBg(accentPalette)}>
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

            <SimpleGrid columns={{ base: 1, md: 3, xl: 1 }} gap="2">
              <DownloadButton label="翻唱人声" href={job.artifacts.vocals} version={artifactVersion} accentPalette={accentPalette} />
              <DownloadButton label="钢琴伴奏" href={job.artifacts.piano} version={artifactVersion} accentPalette={accentPalette} />
              <DownloadButton label="最终混音" href={job.artifacts.final} version={artifactVersion} accentPalette={accentPalette} />
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
  accentPalette,
}: {
  label: string
  href?: string | null
  version: string
  accentPalette: AccentPalette
}) {
  return (
    <Button
      asChild
      size="sm"
      variant="outline"
      colorPalette={accentPalette}
      borderColor={controlSoftBg(accentPalette)}
      color={controlAccentColor(accentPalette)}
      _hover={{ bg: controlSoftBg(accentPalette) }}
      disabled={!href}
      justifyContent="space-between"
    >
      <Link href={href ? artifactUrl(href, version) : undefined} download>
        {label}
        <LuDownload />
      </Link>
    </Button>
  )
}

function apiUrl(path: string) {
  if (path.startsWith("http://") || path.startsWith("https://")) return path
  return `${_apiBase}${path}`
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
