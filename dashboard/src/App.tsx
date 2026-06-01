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
  Progress,
  SimpleGrid,
  Slider,
  Span,
  Spinner,
  Stack,
  Steps,
  Text,
} from "@chakra-ui/react"
import { useEffect, useLayoutEffect, useMemo, useRef, useState } from "react"
import {
  LuAudioLines,
  LuCircleAlert,
  LuHourglass,
  LuMicVocal,
  LuMusic2,
  LuPause,
  LuPiano,
  LuPlay,
  LuRefreshCw,
  LuRocket,
  LuServer,
  LuSlidersHorizontal,
  LuSparkles,
  LuUpload,
  LuVolume2,
  LuX,
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
import avemujicaLogo from "../logo/logo_avemujica.png"
import mygoLogo from "../logo/logo_transparent_bg.png"

import { useColorMode } from "@/components/ui/color-mode"
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

const MYGO_BLUE = "#0888B8"
const MYGO_BLUE_FAINT = "rgba(8, 136, 184, 0.12)"
const MYGO_BLUE_BORDER = "rgba(8, 136, 184, 0.38)"
const MYGO_BLUE_SOFT = "rgba(8, 136, 184, 0.34)"
const DARK_GOTHIC_FONT_STACK = '"Dashboard Alice Gothic", "Dashboard Chinese Gothic", "UnifrakturMaguntia", "Old English Text MT", "Blackletter", "Grenze Gotisch", "华文新魏", "STXinwei", "华文行楷", "STXingkai", "KaiTi", fantasy, serif'
const DARK_GOTHIC_FONT_FACE_CSS = `
  @font-face {
    font-family: "Dashboard Alice Gothic";
    src: url("/fonts/Alice-In-Wonderland/AliceInWonderland-1GzL0-2.ttf") format("truetype");
    font-display: swap;
    size-adjust: 100%;
    unicode-range: U+0000-00FF, U+0100-024F, U+2000-206F;
  }

  @font-face {
    font-family: "Dashboard Chinese Gothic";
    src: url("/fonts/GeTeShiZiTi/GeTeShiZiTi-1.ttf") format("truetype");
    font-display: swap;
    unicode-range: U+2E80-2EFF, U+3000-303F, U+3400-4DBF, U+4E00-9FFF, U+F900-FAFF, U+FF00-FFEF;
  }
`

const DARK_GOTHIC_TEXT_CSS = {
  "& :where(p, button, input, label, [data-part='title'], [data-part='label'], [data-part='value-text'])": {
    fontSize: "1.06em",
  },
}

const MYGO_PALETTE_CSS = {
  "--chakra-colors-color-palette-solid": MYGO_BLUE,
  "--chakra-colors-color-palette-fg": MYGO_BLUE,
  "--chakra-colors-color-palette-focus-ring": MYGO_BLUE,
  "--chakra-colors-color-palette-muted": MYGO_BLUE_FAINT,
  "--chakra-colors-color-palette-subtle": MYGO_BLUE_FAINT,
}

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
  oblivionis: "center center",
  tomorin: "center top",
}

function surfacePanelBg(isDarkMode: boolean) {
  return isDarkMode ? "#000000" : "#ffffff"
}

function surfacePanelBorderColor(isDarkMode: boolean) {
  return isDarkMode ? "rgba(255, 255, 255, 0.16)" : "rgba(0, 0, 0, 0.12)"
}

function surfacePanelInnerBg(isDarkMode: boolean) {
  return isDarkMode ? "#000000" : "#ffffff"
}

function useInitialPanelHeight() {
  const panelRef = useRef<HTMLDivElement | null>(null)
  const [initialHeight, setInitialHeight] = useState<string | null>(null)

  useLayoutEffect(() => {
    if (initialHeight) return

    const panel = panelRef.current
    if (!panel) return

    const measuredHeight = Math.ceil(panel.getBoundingClientRect().height)
    if (measuredHeight > 0) {
      setInitialHeight(`${measuredHeight}px`)
    }
  }, [initialHeight])

  return { panelRef, initialHeight }
}

function controlSolidBg(accentPalette: AccentPalette) {
  return accentPalette === "red" ? "#7f1d1d" : MYGO_BLUE
}

function controlSolidHoverBg(accentPalette: AccentPalette) {
  return accentPalette === "red" ? "#991b1b" : MYGO_BLUE
}

function controlSoftBg(accentPalette: AccentPalette) {
  return accentPalette === "red" ? "#5f151b" : MYGO_BLUE_FAINT
}

function controlAccentColor(accentPalette: AccentPalette) {
  return accentPalette === "red" ? "#b45353" : MYGO_BLUE
}

function App() {
  const { colorMode } = useColorMode()
  const isDarkMode = colorMode === "dark"
  const accentPalette: AccentPalette = isDarkMode ? "red" : "blue"
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
  const jobRoleDetails = config.roles.find((role) => role.id === job?.params.role_id)
  const coverRoleName = jobRoleDetails?.name ?? job?.params.role_id ?? selectedRoleDetails?.name ?? selectedRole
  const isRunning = job?.status === "queued" || job?.status === "running"
  const isProcessing = isRunning || submitting || remixing
  const canSubmit = Boolean(audioFile && selectedRoleDetails?.ready && !isProcessing)
  const canRemix = job?.status === "completed" && Boolean(job.artifacts.final) && !isProcessing
  const eventJobId = job?.job_id
  const eventJobStatus = job?.status
  const audioPreviewUrlRef = useRef<string | null>(null)

  useEffect(() => {
    if (document.getElementById("dark-gothic-fonts")) return

    const fontStyle = document.createElement("style")
    fontStyle.id = "dark-gothic-fonts"
    fontStyle.textContent = DARK_GOTHIC_FONT_FACE_CSS
    document.head.appendChild(fontStyle)
  }, [])

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
    if (isProcessing) return

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
    } catch {
      toaster.create({
        title: "提交失败",
        type: "error",
      })
    } finally {
      setSubmitting(false)
    }
  }

  function resetParameterControls() {
    setKeyShift([selectedRoleDetails?.default_pre_pitch_shift ?? 0])
    setVocalsVolume([config.constraints.vocals_volume.default])
    setPianoVolume([config.constraints.piano_volume.default])
  }

  function handleAudioFileChange(file: File | null) {
    if (isProcessing) return

    if (audioPreviewUrlRef.current) {
      URL.revokeObjectURL(audioPreviewUrlRef.current)
      audioPreviewUrlRef.current = null
    }

    setJob(null)
    setAudioFile(file)
    if (!file) {
      setAudioPreviewUrl(null)
      setSubmitting(false)
      setRemixing(false)
      resetParameterControls()
      return
    }

    const nextPreviewUrl = URL.createObjectURL(file)
    audioPreviewUrlRef.current = nextPreviewUrl
    setAudioPreviewUrl(nextPreviewUrl)
  }

  async function handleRemix() {
    if (isProcessing || !job) return

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
  }

  return (
    <Box minH="100vh" bg={isDarkMode ? "#000000" : "#ffffff"} color="fg" position="relative" overflow="hidden" fontFamily={isDarkMode ? DARK_GOTHIC_FONT_STACK : undefined} css={isDarkMode ? DARK_GOTHIC_TEXT_CSS : undefined}>
      <PullLampColorModeButton />
      <Container maxW="7xl" pt={{ base: 3, md: 4 }} pb={{ base: 5, md: 8 }} position="relative">
        <Stack gap={{ base: 4, md: 5 }}>
          <Grid templateColumns={{ base: "1fr", xl: "1.18fr 0.82fr" }} gap="6" alignItems="stretch">
            <GridItem display="flex" minH="0">
              <Stack gap="6" flex="1" minH="0">
                <WorkflowCard job={job} stages={config.stages} accentPalette={accentPalette} isDarkMode={isDarkMode} />
                <RolePicker
                  roles={config.roles}
                  value={selectedRole}
                  onChange={(roleId) => {
                    const role = config.roles.find((item) => item.id === roleId)
                    setSelectedRole(roleId)
                    setKeyShift([role?.default_pre_pitch_shift ?? 0])
                  }}
                  disabled={isProcessing}
                  stretch
                />
              </Stack>
            </GridItem>

            <GridItem display="flex" minH="0">
              <Stack gap="4" position={{ xl: "sticky" }} top="6" h={{ xl: "100%" }} flex="1" minH="0">
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
                  disabled={isProcessing}
                  onSubmit={handleSubmit}
                  onRemix={handleRemix}
                  accentPalette={accentPalette}
                  isDarkMode={isDarkMode}
                />
                <AudioPanelGroup
                  audioFile={audioFile}
                  previewUrl={audioPreviewUrl}
                  onFileChange={handleAudioFileChange}
                  disabled={isProcessing}
                  job={job}
                  busy={isProcessing}
                  coverTitle={`${coverRoleName}'s cover`}
                  accentPalette={accentPalette}
                  isDarkMode={isDarkMode}
                />
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
          </HStack>
        </Stack>
      </Container>
    </Box>
  )
}

function PullLampColorModeButton() {
  const { colorMode, toggleColorMode } = useColorMode()
  const isDarkMode = colorMode === "dark"
  const dragStartYRef = useRef<number | null>(null)
  const [pullOffset, setPullOffset] = useState(0)

  function getPullOffset(clientY: number) {
    if (dragStartYRef.current === null) return 0
    return Math.min(Math.max(clientY - dragStartYRef.current, 0), 34)
  }

  function releasePull(clientY: number) {
    const nextOffset = getPullOffset(clientY)
    dragStartYRef.current = null
    setPullOffset(0)
    if (nextOffset >= 18) toggleColorMode()
  }

  return (
    <Button
      aria-label={isDarkMode ? "向下拉切换到浅色模式" : "向下拉切换到深色模式"}
      title={isDarkMode ? "向下拉切换到浅色模式" : "向下拉切换到深色模式"}
      position="fixed"
      top="0"
      right={{ base: "4", md: "6" }}
      zIndex="docked"
      w="58px"
      h="104px"
      minW="0"
      p="0"
      variant="plain"
      display="flex"
      alignItems="flex-start"
      justifyContent="center"
      cursor="grab"
      touchAction="none"
      onPointerDown={(event) => {
        dragStartYRef.current = event.clientY
        setPullOffset(0)
        event.currentTarget.setPointerCapture(event.pointerId)
      }}
      onPointerMove={(event) => {
        if (dragStartYRef.current !== null) setPullOffset(getPullOffset(event.clientY))
      }}
      onPointerUp={(event) => releasePull(event.clientY)}
      onPointerCancel={() => {
        dragStartYRef.current = null
        setPullOffset(0)
      }}
      _hover={{ bg: "transparent" }}
      _active={{ cursor: "grabbing" }}
      _focusVisible={{ outline: "2px solid", outlineColor: isDarkMode ? "#b45353" : MYGO_BLUE, outlineOffset: "2px" }}
    >
      <Stack align="center" gap="0">
        <Box
          w="1px"
          h={`${48 + pullOffset}px`}
          bg={isDarkMode ? "rgba(248, 113, 113, 0.46)" : "rgba(8, 136, 184, 0.45)"}
          transition="height 0.18s ease"
        />
        <Stack className="lamp-pull" align="center" gap="0" transition="transform 0.18s ease">
          <Box
            w="10px"
            h="10px"
            rounded="full"
            bg={isDarkMode ? "#6f1d24" : MYGO_BLUE}
            boxShadow={isDarkMode ? "0 0 16px rgba(248, 113, 113, 0.42)" : `0 0 18px ${MYGO_BLUE_SOFT}`}
          />
          <Box
            mt="0"
            w="34px"
            h="24px"
            roundedTop="full"
            roundedBottom="md"
            bg={isDarkMode ? "#1a0508" : "#fff7d6"}
            borderWidth="1px"
            borderColor={isDarkMode ? "rgba(248, 113, 113, 0.30)" : MYGO_BLUE}
            boxShadow={
              isDarkMode
                ? "0 0 0 1px rgba(248, 113, 113, 0.20), 0 0 24px rgba(248, 113, 113, 0.34), 0 10px 24px rgba(127, 29, 29, 0.18)"
                : `0 0 0 1px rgba(8, 136, 184, 0.18), 0 0 22px rgba(250, 204, 21, 0.42), 0 10px 24px rgba(8, 136, 184, 0.10)`
            }
            position="relative"
            _after={{
              content: '""',
              position: "absolute",
              left: "50%",
              bottom: "-8px",
              transform: "translateX(-50%)",
              width: "44px",
              height: "10px",
              borderRadius: "999px",
              background: isDarkMode ? "rgba(248, 113, 113, 0.28)" : "rgba(250, 204, 21, 0.24)",
              filter: "blur(3px)",
            }}
          />
        </Stack>
      </Stack>
    </Button>
  )
}

type WorkflowCardProps = {
  job: Job | null
  stages: BackendConfig["stages"]
  accentPalette: AccentPalette
  isDarkMode: boolean
}

function normalizeDisplayedProgress(job: Job | null) {
  const progress = job?.progress ?? 0
  if (job?.stage_name === "piano_cover" && progress > 55 && progress < 90) {
    return 55
  }

  return progress
}

function WorkflowCard({ job, stages, accentPalette, isDarkMode }: WorkflowCardProps) {
  const { panelRef, initialHeight } = useInitialPanelHeight()
  const activeStep = job?.status === "completed" ? stages.length : Math.max((job?.stage ?? 1) - 1, 0)
  const progress = normalizeDisplayedProgress(job)
  const running = job?.status === "running" || job?.status === "queued"

  return (
    <Card.Root
      ref={panelRef}
      h={initialHeight ?? undefined}
      minH={initialHeight ?? undefined}
      maxH={initialHeight ?? undefined}
      overflow="hidden"
      bg={surfacePanelBg(isDarkMode)}
      borderWidth="1px"
      borderColor={surfacePanelBorderColor(isDarkMode)}
      shadow="sm"
    >
      <Card.Header pb="3" flexShrink="0">
        <Grid templateColumns="24px 1fr 24px" alignItems="center">
          <Box />
          <Card.Title textAlign="center" fontSize={{ base: "3xl", md: "4xl" }}>四阶段处理进度</Card.Title>
          <Box display="flex" justifyContent="flex-end">
            {running && <Spinner size="sm" color={controlAccentColor(accentPalette)} />}
          </Box>
        </Grid>
      </Card.Header>
      <Card.Body
        bg={surfacePanelBg(isDarkMode)}
        minH="0"
        overflowY="auto"
        display="flex"
        alignItems="center"
        justifyContent="center"
      >
        <Stack w="100%" maxW="720px" gap="5" align="stretch" justify="center">
          <Steps.Root step={activeStep} count={stages.length} colorPalette={accentPalette} size="sm" css={accentPalette === "blue" ? MYGO_PALETTE_CSS : undefined}>
            <Steps.List>
              {stages.map((stage, index) => {
                const StageIcon = STAGE_ICONS[index] ?? LuSparkles
                const stepHighlighted = index <= activeStep || job?.status === "completed"
                const stepComplete = index < activeStep || job?.status === "completed"

                return (
                  <Steps.Item key={stage.name} index={index} title={stage.label}>
                    <Steps.Indicator
                      bg={accentPalette === "blue" && stepHighlighted ? MYGO_BLUE : undefined}
                      borderColor={accentPalette === "blue" && stepHighlighted ? MYGO_BLUE : undefined}
                      color={accentPalette === "blue" && stepHighlighted ? "white" : undefined}
                    >
                      <Steps.Status complete={<LuCheck />} incomplete={<StageIcon />} />
                    </Steps.Indicator>
                    <Steps.Title
                      display={{ base: "none", md: "block" }}
                      fontSize="md"
                      color={accentPalette === "blue" && stepHighlighted ? MYGO_BLUE : undefined}
                    >
                      {STAGE_COPY[stage.name] ?? stage.label}
                    </Steps.Title>
                    <Steps.Separator bg={accentPalette === "blue" && stepComplete ? MYGO_BLUE : undefined} />
                  </Steps.Item>
                )
              })}
            </Steps.List>
          </Steps.Root>

          <Progress.Root value={progress} colorPalette={accentPalette} striped animated={running} css={accentPalette === "blue" ? MYGO_PALETTE_CSS : undefined}>
            <HStack justify="space-between" mb="1">
              <Progress.Label color="fg.muted" fontSize="md">
                {job?.stage_name ? STAGE_COPY[job.stage_name] ?? job.stage_name : "等待提交"}
              </Progress.Label>
              <Progress.ValueText>{progress}%</Progress.ValueText>
            </HStack>
            <Progress.Track>
              <Progress.Range
                bg={accentPalette === "red" ? "#7f1d1d" : MYGO_BLUE}
                backgroundImage="linear-gradient(45deg, rgba(255, 255, 255, 0.22) 25%, transparent 25%, transparent 50%, rgba(255, 255, 255, 0.22) 50%, rgba(255, 255, 255, 0.22) 75%, transparent 75%, transparent)"
                backgroundSize="1rem 1rem"
              />
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
        </Stack>
      </Card.Body>
    </Card.Root>
  )
}

type AudioPanelGroupProps = {
  audioFile: File | null
  previewUrl: string | null
  onFileChange: (file: File | null) => void
  disabled: boolean
  job: Job | null
  busy: boolean
  coverTitle: string
  accentPalette: AccentPalette
  isDarkMode: boolean
}

function AudioPanelGroup({
  audioFile,
  previewUrl,
  onFileChange,
  disabled,
  job,
  busy,
  coverTitle,
  accentPalette,
  isDarkMode,
}: AudioPanelGroupProps) {
  return (
    <Box
      display="grid"
      gridTemplateRows="minmax(0, 1fr) minmax(0, 1fr)"
      gap="4"
      flex={{ xl: "1" }}
      minH={{ base: "360px", xl: "0" }}
    >
      <UploadCard
        audioFile={audioFile}
        previewUrl={previewUrl}
        onFileChange={onFileChange}
        disabled={disabled}
        accentPalette={accentPalette}
        isDarkMode={isDarkMode}
        stretch
      />
      <ResultCard
        job={job}
        audioFile={audioFile}
        busy={busy}
        coverTitle={coverTitle}
        accentPalette={accentPalette}
        isDarkMode={isDarkMode}
        stretch
      />
    </Box>
  )
}

type UploadCardProps = {
  audioFile: File | null
  previewUrl: string | null
  onFileChange: (file: File | null) => void
  disabled: boolean
  accentPalette: AccentPalette
  isDarkMode: boolean
  stretch?: boolean
}

function UploadCard({ audioFile, previewUrl, onFileChange, disabled, accentPalette, isDarkMode, stretch = false }: UploadCardProps) {
  const { panelRef, initialHeight } = useInitialPanelHeight()
  const panelHeight = stretch ? "100%" : initialHeight ?? undefined
  const panelMinHeight = stretch ? "0" : initialHeight ?? undefined
  const panelMaxHeight = stretch ? undefined : initialHeight ?? undefined

  return (
    <Card.Root
      ref={panelRef}
      h={panelHeight}
      minH={panelMinHeight}
      maxH={panelMaxHeight}
      overflow="hidden"
      bg={surfacePanelBg(isDarkMode)}
      borderWidth="1px"
      borderColor={surfacePanelBorderColor(isDarkMode)}
      shadow="sm"
    >
      <Card.Header py="3" pb="2" flexShrink="0">
        <HStack justify="space-between" gap="3">
          <Card.Title>上传原唱</Card.Title>
          {audioFile && (
            <Button
              aria-label="取消上传"
              size="xs"
              variant="ghost"
              colorPalette={accentPalette}
              color={controlAccentColor(accentPalette)}
              _hover={{ bg: controlSoftBg(accentPalette) }}
              disabled={disabled}
              flexShrink="0"
              onClick={() => onFileChange(null)}
            >
              <LuX />
            </Button>
          )}
        </HStack>
      </Card.Header>
      <Card.Body pt="2" bg={surfacePanelBg(isDarkMode)} minH="0" overflow="hidden" display="flex">
        <FileUpload.Root
          accept={["audio/*"]}
          maxFiles={1}
          disabled={disabled}
          onFileChange={(details) => onFileChange(details.acceptedFiles[0] ?? null)}
          alignItems="stretch"
          w="100%"
          h="100%"
        >
          <FileUpload.HiddenInput />
          <FileUpload.Dropzone
            h="100%"
            minH="0"
            w="100%"
            boxSizing="border-box"
            py={audioFile ? "0" : "3"}
            px={audioFile ? "0" : "3"}
            borderWidth={audioFile ? "0" : "1px"}
            borderStyle="dashed"
            borderColor={surfacePanelBorderColor(isDarkMode)}
            rounded="md"
            bg={audioFile ? "transparent" : surfacePanelInnerBg(isDarkMode)}
          >
            {audioFile ? (
              <HStack
                w="100%"
                h="100%"
                gap="2"
                onClick={(event) => event.stopPropagation()}
                onPointerDown={(event) => event.stopPropagation()}
              >
                <Box flex="1" minW="0" h="100%">
                  {previewUrl && (
                    <AudioPlayer
                      src={previewUrl}
                      title="Original"
                      accentPalette={accentPalette}
                      isDarkMode={isDarkMode}
                      fill
                    />
                  )}
                </Box>
              </HStack>
            ) : (
              <>
                <Icon fontSize="xl" color={controlAccentColor(accentPalette)}>
                  <LuUpload />
                </Icon>
                <FileUpload.DropzoneContent>
                  <Text fontWeight="semibold" textStyle="sm">拖拽音频到这里，或点击选择文件</Text>
                </FileUpload.DropzoneContent>
              </>
            )}
          </FileUpload.Dropzone>
        </FileUpload.Root>
      </Card.Body>
    </Card.Root>
  )
}

type RolePickerProps = {
  roles: Role[]
  value: string
  onChange: (value: string) => void
  disabled: boolean
  stretch?: boolean
}

function RolePicker({ roles, value, onChange, disabled, stretch = false }: RolePickerProps) {
  const { panelRef, initialHeight } = useInitialPanelHeight()
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
      ref={panelRef}
      h={stretch ? { xl: "100%" } : initialHeight ?? undefined}
      minH={stretch ? { xl: "0" } : initialHeight ?? undefined}
      maxH={stretch ? undefined : initialHeight ?? undefined}
      flex={stretch ? { xl: "1" } : undefined}
      overflow="hidden"
      bg={surfacePanelBg(isDarkMode)}
      borderWidth="1px"
      borderColor={surfacePanelBorderColor(isDarkMode)}
      shadow="sm"
    >
      <Card.Header pb="3" flexShrink="0">
        <Box position="relative" h={{ base: "50px", md: "62px" }} overflow="visible">
          <Image
            src={isDarkMode ? avemujicaLogo : mygoLogo}
            alt={isDarkMode ? "Ave Mujica" : "MyGO"}
            position="absolute"
            left="50%"
            top="56%"
            transform="translate(-50%, -50%)"
            h={isDarkMode ? { base: "72px", md: "88px" } : { base: "98px", md: "120px" }}
            maxW={isDarkMode ? "390px" : "520px"}
            objectFit="contain"
          />
        </Box>
      </Card.Header>
      <Card.Body bg={surfacePanelBg(isDarkMode)} minH="0" overflowY="auto">
        <SimpleGrid columns={{ base: 1, md: 2, xl: 4 }} gap={{ base: 3, md: 4 }}>
          {visibleRoles.map((role) => {
            const roleId = role.id as CharacterRoleId
            const selected = role.id === value
            const roleImage = CHARACTER_ROLE_IMAGES[roleId]
            const cardDisabled = disabled || !role.ready
            const nameFontSize = roleId === "oblivionis" ? { base: "xl", md: "2xl" } : { base: "2xl", md: "3xl" }

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
                borderWidth={selected ? "4px" : "2px"}
                borderColor={
                  selected
                    ? isDarkMode ? "#b45353" : MYGO_BLUE
                    : isDarkMode ? "rgba(248, 113, 113, 0.30)" : MYGO_BLUE_BORDER
                }
                bg={
                  isDarkMode
                    ? "linear-gradient(145deg, #3f0b12 0%, #68131b 52%, #1f0509 100%)"
                    : "linear-gradient(145deg, #d9eef7 0%, #eef7f9 50%, #e9eefb 100%)"
                }
                boxShadow={
                  selected
                    ? isDarkMode ? "0 18px 42px rgba(248, 113, 113, 0.24)" : `0 18px 40px ${MYGO_BLUE_SOFT}`
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
                        borderColor: selected ? isDarkMode ? "#b45353" : MYGO_BLUE : isDarkMode ? "#b45353" : MYGO_BLUE_BORDER,
                      }
                }
                _focusVisible={{
                  outline: "3px solid",
                  outlineColor: isDarkMode ? "#b45353" : MYGO_BLUE,
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
                  align="center"
                  gap="1"
                  px="3"
                  py="3"
                  overflow="hidden"
                  textAlign="center"
                  rounded="xl"
                  bg={isDarkMode ? "rgba(37, 7, 10, 0.82)" : "rgba(255, 255, 255, 0.84)"}
                  color={isDarkMode ? "red.50" : "gray.800"}
                  boxShadow={isDarkMode ? "0 10px 30px rgba(0, 0, 0, 0.34)" : "0 10px 28px rgba(15, 23, 42, 0.14)"}
                  backdropFilter="blur(8px)"
                >
                  <Text
                    maxW="100%"
                    fontWeight="bold"
                    fontSize={nameFontSize}
                    color={isDarkMode ? "red.100" : MYGO_BLUE}
                    lineHeight="1.25"
                    whiteSpace="nowrap"
                    overflow="hidden"
                    textOverflow="ellipsis"
                    pb="0.5"
                  >
                    {role.name}
                  </Text>
                  {!role.ready && (
                    <Text maxW="100%" color={isDarkMode ? "orange.200" : "orange.700"} textStyle="xs" lineClamp={2}>
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
  disabled: boolean
  onSubmit: () => void
  onRemix: () => void
  accentPalette: AccentPalette
  isDarkMode: boolean
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
    disabled,
    onSubmit,
    onRemix,
    accentPalette,
    isDarkMode,
  } = props

  return (
    <Card.Root
      overflow="visible"
      bg={surfacePanelBg(isDarkMode)}
      borderWidth="1px"
      borderColor={surfacePanelBorderColor(isDarkMode)}
      shadow="sm"
      minH={{ xl: "440px" }}
      display="flex"
      flexDir="column"
    >
      <Card.Header py="3" pb="2" flexShrink="0">
        <HStack justify="space-between">
          <Stack gap="1">
            <Card.Title>参数控制</Card.Title>
          </Stack>
          <Icon color={controlAccentColor(accentPalette)} fontSize="xl">
            <LuSlidersHorizontal />
          </Icon>
        </HStack>
      </Card.Header>
      <Card.Body pt="1" pb="2" gap="3" bg={surfacePanelBg(isDarkMode)} overflow="visible" flex="1" display="flex">
        <Stack gap="3" flex="1" justify="space-evenly">
          <ParameterSlider
            label="升降 Key"
            icon={<LuMusic2 style={{ transform: "translateX(1.5px)" }} />}
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
            disabled={disabled}
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
            disabled={disabled}
          />
          <VolumeSlider
            label="钢琴伴奏音量"
            value={pianoVolume}
            min={constraints.piano_volume.min}
            max={constraints.piano_volume.max}
            step={constraints.piano_volume.step}
            onChange={onPianoVolumeChange}
            icon={<LuPiano />}
            accentPalette={accentPalette}
            disabled={disabled}
          />
        </Stack>
      </Card.Body>
      <Card.Footer pt="1" flexDir="column" alignItems="stretch" gap="2" bg={surfacePanelBg(isDarkMode)} flexShrink="0">
        <Button
          size="sm"
          colorPalette={accentPalette}
          bg={controlSolidBg(accentPalette)}
          _hover={{ bg: controlSolidHoverBg(accentPalette) }}
          disabled={disabled || !canSubmit}
          loading={submitting}
          onClick={onSubmit}
        >
          <LuRocket />
          提交完整任务
        </Button>
        <Button
          size="sm"
          variant="outline"
          colorPalette={accentPalette}
          borderColor={controlSoftBg(accentPalette)}
          color={controlAccentColor(accentPalette)}
          _hover={{ bg: controlSoftBg(accentPalette) }}
          disabled={disabled || !canRemix}
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
  disabled: boolean
  onChange: (value: number[]) => void
}

function VolumeSlider({ label, value, min, max, step, icon, accentPalette, disabled, onChange }: VolumeSliderProps) {
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
      disabled={disabled}
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
  disabled?: boolean
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
  disabled = false,
  onChange,
}: ParameterSliderProps) {
  return (
    <Slider.Root
      css={colorPalette === "blue" ? MYGO_PALETTE_CSS : undefined}
      value={value}
      min={min}
      max={max}
      step={step}
      origin={origin}
      colorPalette={colorPalette}
      aria-disabled={disabled}
      opacity="1"
      thumbAlignment="center"
      getAriaValueText={(details) => String(details.value)}
      onValueChange={(details) => {
        if (!disabled) onChange(details.value)
      }}
    >
      <HStack justify="space-between" mb="1">
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
      <Slider.Control data-has-mark-label pointerEvents={disabled ? "none" : undefined} opacity={disabled ? 0.72 : 1}>
        <Slider.Track bg="bg.muted">
          <Slider.Range bg={controlSolidBg(colorPalette)} />
        </Slider.Track>
        <Slider.Thumb
          index={0}
          bg={colorPalette === "red" ? "#2d070b" : "#ffffff"}
          borderColor={controlSolidBg(colorPalette)}
          boxShadow={colorPalette === "red" ? "0 0 0 3px rgba(127, 29, 29, 0.24)" : `0 0 0 3px ${MYGO_BLUE_FAINT}`}
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
  audioFile: File | null
  busy: boolean
  coverTitle: string
  accentPalette: AccentPalette
  isDarkMode: boolean
  stretch?: boolean
}

type ResultStatus = {
  label: string
  colorPalette: string
}

function getResultStatus(job: Job | null, audioFile: File | null, busy: boolean, accentPalette: AccentPalette): ResultStatus {
  if (busy || job?.status === "queued" || job?.status === "running") {
    return { label: "Running", colorPalette: accentPalette }
  }

  if (job?.status === "completed") return { label: "Completed", colorPalette: "green" }
  if (job?.status === "failed") return { label: "Failed", colorPalette: "red" }
  if (job?.status === "cancelled") return { label: "Cancelled", colorPalette: "gray" }
  if (audioFile) return { label: "Uploaded", colorPalette: accentPalette }

  return { label: "Idle", colorPalette: "gray" }
}

function ResultCard({ job, audioFile, busy, coverTitle, accentPalette, isDarkMode, stretch = false }: ResultCardProps) {
  const { panelRef, initialHeight } = useInitialPanelHeight()
  const artifactVersion = job?.updated_at ?? `${job?.progress ?? 0}`
  const finalMix = job?.artifacts.final
  const status = getResultStatus(job, audioFile, busy, accentPalette)
  const statusUsesAccent = status.colorPalette === accentPalette
  const panelHeight = stretch ? "100%" : initialHeight ?? undefined
  const panelMinHeight = stretch ? "0" : initialHeight ?? undefined
  const panelMaxHeight = stretch ? undefined : initialHeight ?? undefined

  return (
    <Card.Root
      ref={panelRef}
      h={panelHeight}
      minH={panelMinHeight}
      maxH={panelMaxHeight}
      overflow="hidden"
      bg={surfacePanelBg(isDarkMode)}
      borderWidth="1px"
      borderColor={surfacePanelBorderColor(isDarkMode)}
      shadow="sm"
    >
      <Card.Header py="3" pb="2" flexShrink="0">
        <HStack justify="space-between" gap="3">
          <Card.Title>钢伴翻唱</Card.Title>
          <Badge
            colorPalette={status.colorPalette}
            variant="surface"
            bg={statusUsesAccent ? controlSoftBg(accentPalette) : undefined}
            color={statusUsesAccent ? controlAccentColor(accentPalette) : undefined}
            borderColor={statusUsesAccent ? controlAccentColor(accentPalette) : undefined}
          >
            {status.label}
          </Badge>
        </HStack>
      </Card.Header>
      <Card.Body pt="2" bg={surfacePanelBg(isDarkMode)} minH="0" overflow="hidden" display="flex">
        {finalMix ? (
          <AudioPlayer
            key={artifactVersion}
            src={artifactUrl(finalMix, artifactVersion)}
            title={coverTitle}
            accentPalette={accentPalette}
            isDarkMode={isDarkMode}
            fill
          />
        ) : (
          <OutputPlaceholder
            audioFile={audioFile}
            busy={busy}
            accentPalette={accentPalette}
            isDarkMode={isDarkMode}
          />
        )}
      </Card.Body>
    </Card.Root>
  )
}

type OutputPlaceholderProps = {
  audioFile: File | null
  busy: boolean
  accentPalette: AccentPalette
  isDarkMode: boolean
}

function OutputPlaceholder({ audioFile, busy, accentPalette, isDarkMode }: OutputPlaceholderProps) {
  const waiting = busy
  const PlaceholderIcon = waiting ? LuHourglass : LuCircleAlert
  const message = waiting
    ? "翻唱生成中，请耐心等待"
    : audioFile
      ? "暂无输出音频，请提交任务"
      : "暂无原唱，请上传原唱"

  return (
    <Box
      h="100%"
      minH="0"
      w="100%"
      display="flex"
      alignItems="center"
      justifyContent="center"
      rounded="md"
      borderWidth="1px"
      borderStyle="dashed"
      borderColor={surfacePanelBorderColor(isDarkMode)}
      bg={surfacePanelInnerBg(isDarkMode)}
      px="3"
      py="3"
    >
      <Stack align="center" gap="2" textAlign="center">
        <Icon fontSize="2xl" color={controlAccentColor(accentPalette)}>
          <PlaceholderIcon />
        </Icon>
        <Text color="fg.muted" textStyle="sm" fontWeight="semibold">
          {message}
        </Text>
      </Stack>
    </Box>
  )
}

type AudioPlayerProps = {
  src: string
  title: string
  accentPalette: AccentPalette
  isDarkMode: boolean
  fill?: boolean
}

function AudioPlayer({ src, title, accentPalette, isDarkMode, fill = false }: AudioPlayerProps) {
  const audioRef = useRef<HTMLAudioElement | null>(null)
  const [playing, setPlaying] = useState(false)
  const [duration, setDuration] = useState(0)
  const [currentTime, setCurrentTime] = useState(0)
  const playable = duration > 0

  async function togglePlayback() {
    const audio = audioRef.current
    if (!audio) return

    if (audio.paused) {
      await audio.play()
      setPlaying(true)
    } else {
      audio.pause()
      setPlaying(false)
    }
  }

  function handleSeek(value: number[]) {
    const nextTime = value[0] ?? 0
    const audio = audioRef.current
    if (audio) audio.currentTime = nextTime
    setCurrentTime(nextTime)
  }

  return (
    <Box
      w="100%"
      h={fill ? "100%" : undefined}
      p="3"
      display="flex"
      alignItems="center"
      rounded="md"
      borderWidth="1px"
      borderColor={isDarkMode ? "rgba(248, 113, 113, 0.26)" : "rgba(8, 136, 184, 0.18)"}
      bg={isDarkMode ? "#070707" : "#f8fbff"}
      boxShadow={isDarkMode ? "inset 0 1px 0 rgba(255, 255, 255, 0.06)" : "inset 0 1px 0 rgba(255, 255, 255, 0.9)"}
    >
      <audio
        ref={audioRef}
        src={src}
        preload="metadata"
        onLoadedMetadata={(event) => {
          const nextDuration = event.currentTarget.duration
          setDuration(Number.isFinite(nextDuration) ? nextDuration : 0)
        }}
        onTimeUpdate={(event) => setCurrentTime(event.currentTarget.currentTime)}
        onEnded={() => setPlaying(false)}
      />
      <HStack w="100%" gap="3" align="center">
        <Button
          aria-label={playing ? "暂停音频" : "播放音频"}
          size="sm"
          w="38px"
          h="38px"
          p="0"
          rounded="full"
          colorPalette={accentPalette}
          bg={controlSolidBg(accentPalette)}
          _hover={{ bg: controlSolidHoverBg(accentPalette) }}
          onClick={togglePlayback}
        >
          {playing ? <LuPause /> : <LuPlay />}
        </Button>
        <Box flex="1" minW="0">
          <HStack justify="space-between" gap="3" mb="1">
            <HStack gap="2" minW="0">
              <Icon color={controlAccentColor(accentPalette)} flexShrink="0">
                <LuMusic2 />
              </Icon>
              <Text fontWeight="semibold" textStyle="sm" truncate>
                {title}
              </Text>
            </HStack>
            <Text color="fg.muted" textStyle="xs" flexShrink="0">
              {formatDuration(currentTime)} / {formatDuration(duration)}
            </Text>
          </HStack>
          <Slider.Root
            css={accentPalette === "blue" ? MYGO_PALETTE_CSS : undefined}
            value={[currentTime]}
            min={0}
            max={duration || 1}
            step={0.1}
            colorPalette={accentPalette}
            disabled={!playable}
            onValueChange={(details) => handleSeek(details.value)}
          >
            <Slider.Control>
              <Slider.Track bg={isDarkMode ? "rgba(255, 255, 255, 0.12)" : "rgba(15, 23, 42, 0.10)"}>
                <Slider.Range bg={controlSolidBg(accentPalette)} />
              </Slider.Track>
              <Slider.Thumb
                index={0}
                bg={isDarkMode ? "#111111" : "#ffffff"}
                borderColor={controlSolidBg(accentPalette)}
              >
                <Slider.HiddenInput />
              </Slider.Thumb>
            </Slider.Control>
          </Slider.Root>
        </Box>
      </HStack>
    </Box>
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

function formatDuration(value: number) {
  if (!Number.isFinite(value) || value <= 0) return "0:00"

  const minutes = Math.floor(value / 60)
  const seconds = Math.floor(value % 60)
  return `${minutes}:${String(seconds).padStart(2, "0")}`
}

function formatSigned(value: number) {
  return value > 0 ? `+${value}` : String(value)
}

export default App
