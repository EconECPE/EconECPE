import { Badge, Button, Card, Group, Stack, Text, Tooltip } from "@mantine/core";
import { IconArrowBackUp, IconCheck } from "@tabler/icons-react";

// Every distinct pair proposed by any model, collapsed across sources and
// ranked by how many back it. This is the fast path: one click here beats
// reading nine per-model panels that mostly repeat each other.
export default function ProposalPanel({ proposals, onAccept, onAcceptAll, onNeutral }) {
  if (!proposals) return null;
  const { proposals: items, nAnswered, nSaidNeutral, nSources } = proposals;

  const consensusNeutral = nAnswered > 0 && nSaidNeutral * 2 > nAnswered;

  return (
    <Stack gap="xs">
      <Group justify="space-between">
        <Text size="xs" fw={700} c="dimmed" tt="uppercase">
          All model proposals ({nAnswered}/{nSources} answered)
        </Text>
        {items.length > 1 && (
          <Button size="compact-xs" variant="light" color="teal"
                  leftSection={<IconCheck size={12} />}
                  onClick={() => onAcceptAll(items)}>
            Use all {items.length}
          </Button>
        )}
      </Group>

      {nSaidNeutral > 0 && (
        <Group gap={6}>
          <Badge size="sm" color={consensusNeutral ? "orange" : "gray"} variant="light">
            {nSaidNeutral}/{nAnswered} said neutral
          </Badge>
          <Button size="compact-xs" variant="subtle" color="orange" onClick={onNeutral}>
            Mark neutral
          </Button>
        </Group>
      )}

      {items.length === 0 && (
        <Text size="sm" c="dimmed">No model proposed any pair for this comment.</Text>
      )}

      {items.map((p, i) => (
        <Card key={i} withBorder padding="xs" radius="md"
              bg={p.support >= 3 ? "var(--mantine-color-teal-light)"
                                 : "var(--mantine-color-blue-light)"}>
          <Stack gap={4}>
            <Group gap={6}>
              <Tooltip label={p.backers.join(", ")} multiline w={260} withArrow>
                <Badge size="sm" variant="filled"
                       color={p.support >= 3 ? "teal" : p.support >= 2 ? "blue" : "gray"}>
                  {p.support}× backed
                </Badge>
              </Tooltip>
              <Badge size="sm">{p.emotion}</Badge>
              <Badge size="sm" variant="outline">{p.cause_category}</Badge>
              {p.sarcasm && (
                <Badge size="sm" variant="outline" color="grape">sarcasm</Badge>
              )}
              <Badge size="sm" variant="outline">intensity {p.intensity}</Badge>
              {p.target_asset !== "none" && (
                <Badge size="sm" variant="outline" color="indigo">{p.target_asset}</Badge>
              )}
            </Group>
            <Text size="xs"><b>emotion:</b> “{p.emotion_span}”</Text>
            <Text size="xs">
              <b>cause{p.cause_source ? ` (${p.cause_source})` : ""}:</b>{" "}
              {p.cause_span ? `“${p.cause_span}”` : "—"}
            </Text>
            <Text size="xs" c="dimmed">backed by: {p.backers.join(", ")}</Text>
            <Button size="compact-xs" variant="light"
                    leftSection={<IconArrowBackUp size={12} />}
                    onClick={() => onAccept(p)}>
              Use as my pair
            </Button>
          </Stack>
        </Card>
      ))}
    </Stack>
  );
}
