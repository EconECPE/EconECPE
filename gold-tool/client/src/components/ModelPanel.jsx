import { Alert, Badge, Button, Card, Group, Stack, Text } from "@mantine/core";
import { IconArrowBackUp } from "@tabler/icons-react";

const STATUS_INFO = {
  pending: { label: "Model hasn't labeled this comment yet", color: "gray" },
  raw_unvalidated: { label: "Model output (not yet schema-validated)", color: "yellow" },
  validated: { label: "Model output (validated)", color: "teal" },
  validated_repaired: { label: "Model output (validated after repair)", color: "teal" },
  invalid: { label: "Model output failed validation", color: "red" },
  error: { label: "Model call errored", color: "red" },
};

export default function ModelPanel({ model, onAccept }) {
  const info = STATUS_INFO[model.status] || STATUS_INFO.pending;

  return (
    <Stack gap="xs">
      <Group justify="space-between">
        <Text size="xs" fw={700} c="dimmed" tt="uppercase">
          {model.model}
        </Text>
        <Badge size="sm" color={info.color} variant="light">
          {info.label}
        </Badge>
      </Group>

      {model.errors?.length > 0 && (
        <Alert color="red" p="xs">
          {model.errors.join("; ")}
        </Alert>
      )}

      {model.status !== "pending" && model.pairs.length === 0 && (
        <Text size="sm" c="dimmed">
          Model said: neutral (no pairs).
        </Text>
      )}

      {model.pairs.map((p, i) => (
        <Card key={i} withBorder padding="xs" radius="md" bg="var(--mantine-color-blue-light)">
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
              {/* consensus-only fields: raw per-model pairs don't have these */}
              {typeof p.support === "number" && (
                <Badge size="sm" variant="dot" color={p.support >= 2 ? "teal" : "gray"}>
                  support {p.support}/{p.models?.length ?? "?"}
                </Badge>
              )}
              {p.agreement && Object.values(p.agreement).includes("judge") && (
                <Badge size="sm" variant="dot" color="orange">
                  judge-decided
                </Badge>
              )}
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
            <Button
              size="compact-xs"
              variant="light"
              leftSection={<IconArrowBackUp size={12} />}
              onClick={() => onAccept(p)}
            >
              Use as my pair
            </Button>
          </Stack>
        </Card>
      ))}
    </Stack>
  );
}
