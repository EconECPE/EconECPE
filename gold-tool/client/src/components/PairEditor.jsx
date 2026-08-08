import { Card, Group, Select, Text, Slider, Switch, ActionIcon, Stack, Button, Box } from "@mantine/core";
import { IconTrash, IconHighlight } from "@tabler/icons-react";

function captureSelection() {
  const sel = window.getSelection();
  if (!sel || sel.isCollapsed) return null;
  const t = sel.toString().trim();
  return t || null;
}

function SpanField({ label, value, disabled, onCapture, onClear }) {
  return (
    <Box>
      <Group justify="space-between" mb={2}>
        <Text size="xs" fw={600} c="dimmed">{label}</Text>
        <Group gap={4}>
          <Button
            size="compact-xs"
            variant="light"
            leftSection={<IconHighlight size={12} />}
            disabled={disabled}
            onClick={onCapture}
          >
            Capture selection
          </Button>
          {value && (
            <ActionIcon size="sm" variant="subtle" color="gray" onClick={onClear}>
              ×
            </ActionIcon>
          )}
        </Group>
      </Group>
      <Box
        p={6}
        style={{
          minHeight: 32,
          border: "1px solid var(--mantine-color-gray-4)",
          borderRadius: 4,
          background: disabled ? "var(--mantine-color-gray-1)" : undefined,
        }}
      >
        <Text size="sm" c={value ? undefined : "dimmed"} style={{ whiteSpace: "pre-wrap" }}>
          {value || "highlight text on the left, then click Capture selection"}
        </Text>
      </Box>
    </Box>
  );
}

export default function PairEditor({ index, pair, schema, onChange, onRemove }) {
  const noCause = pair.cause_span === null;

  function set(patch) {
    onChange(index, patch);
  }

  return (
    <Card withBorder padding="sm" radius="md">
      <Group justify="space-between" mb="xs">
        <Text size="xs" fw={700} c="dimmed">
          PAIR {index + 1}
        </Text>
        <ActionIcon color="red" variant="subtle" size="sm" onClick={() => onRemove(index)}>
          <IconTrash size={14} />
        </ActionIcon>
      </Group>

      <Stack gap="sm">
        <Select
          label="Emotion"
          placeholder="choose…"
          data={schema.emotions}
          value={pair.emotion || null}
          onChange={(v) => set({ emotion: v })}
          size="xs"
        />

        <SpanField
          label="Emotion span (verbatim, from the COMMENT only)"
          value={pair.emotion_span}
          onCapture={() => {
            const t = captureSelection();
            if (t) set({ emotion_span: t });
          }}
          onClear={() => set({ emotion_span: "" })}
        />

        <Switch
          size="xs"
          label="No identifiable cause"
          checked={noCause}
          onChange={(e) =>
            set(
              e.currentTarget.checked
                ? { cause_span: null, cause_category: "unclear" }
                : { cause_span: "", cause_category: "" },
            )
          }
        />

        <SpanField
          label="Cause span (verbatim, from comment / parent / post title)"
          value={pair.cause_span}
          disabled={noCause}
          onCapture={() => {
            const t = captureSelection();
            if (t) set({ cause_span: t });
          }}
          onClear={() => set({ cause_span: "" })}
        />

        <Select
          label="Cause category"
          placeholder="choose…"
          data={noCause ? ["unclear"] : schema.causeCategories.filter((c) => c !== "unclear")}
          value={pair.cause_category || null}
          onChange={(v) => set({ cause_category: v })}
          disabled={noCause}
          size="xs"
        />

        <Select
          label="Target asset"
          data={schema.targetAssets}
          value={pair.target_asset || null}
          onChange={(v) => set({ target_asset: v })}
          size="xs"
        />

        <Box>
          <Text size="xs" fw={600} c="dimmed" mb={4}>
            Intensity: {pair.intensity?.toFixed(2)}
          </Text>
          <Slider
            size="sm"
            min={0}
            max={1}
            step={0.05}
            value={pair.intensity}
            onChange={(v) => set({ intensity: v })}
            marks={schema.intensityAnchors.map((a) => ({ value: a, label: String(a) }))}
          />
        </Box>

        <Switch
          size="xs"
          label="Sarcasm (surface text is ironic — label the intended emotion above)"
          checked={!!pair.sarcasm}
          onChange={(e) => set({ sarcasm: e.currentTarget.checked })}
        />
      </Stack>
    </Card>
  );
}
