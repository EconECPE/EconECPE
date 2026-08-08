import { Badge, Button, Card, Group, Stack, Text } from "@mantine/core";
import { IconArrowBackUp } from "@tabler/icons-react";

// Read-only rendering of one annotator's independent call on a comment, with
// an optional "use this pair" hook for building the custom adjudicated call.
export default function AnnotatorPanel({ label, record, onUsePair }) {
  const pairs = record?.pairs || [];
  return (
    <Stack gap="xs">
      <Group justify="space-between">
        <Text size="xs" fw={700} c="dimmed" tt="uppercase">
          {label}
        </Text>
        {record?.neutral && (
          <Badge size="sm" color="gray" variant="light">
            neutral
          </Badge>
        )}
      </Group>

      {!record && (
        <Text size="sm" c="dimmed">
          No label found.
        </Text>
      )}
      {record && record.neutral && pairs.length === 0 && (
        <Text size="sm" c="dimmed">
          Called neutral — no economically-caused emotion.
        </Text>
      )}

      {pairs.map((p, i) => (
        <Card key={i} withBorder padding="xs" radius="md">
          <Stack gap={4}>
            <Group gap={6}>
              <Badge size="sm">{p.emotion}</Badge>
              {p.sarcasm && (
                <Badge size="sm" variant="outline" color="grape">
                  sarcasm
                </Badge>
              )}
              <Badge size="sm" variant="outline">
                intensity {p.intensity}
              </Badge>
            </Group>
            <Text size="xs">
              <b>emotion:</b> “{p.emotion_span}”
            </Text>
            <Text size="xs">
              <b>cause ({p.cause_category}{p.cause_source ? `, ${p.cause_source}` : ""}):</b>{" "}
              {p.cause_span ? `“${p.cause_span}”` : "—"}
            </Text>
            <Text size="xs">
              <b>target:</b> {p.target_asset}
            </Text>
            {onUsePair && (
              <Button
                size="compact-xs"
                variant="light"
                leftSection={<IconArrowBackUp size={12} />}
                onClick={() => onUsePair(p)}
              >
                Use this pair
              </Button>
            )}
          </Stack>
        </Card>
      ))}
    </Stack>
  );
}
