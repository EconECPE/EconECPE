import { useState } from "react";
import { ScrollArea, Stack, UnstyledButton, Group, Text, Badge, SegmentedControl } from "@mantine/core";

const STRATUM_COLORS = { random: "gray", topic: "blue", emotive: "grape" };

export default function CommentList({ comments, selectedId, onSelect, doneCount }) {
  const [filter, setFilter] = useState("all");

  const visible = comments.filter((c) => {
    if (filter === "done") return c.done;
    if (filter === "todo") return !c.done;
    return true;
  });

  return (
    <Stack h="100%" gap="xs">
      <Text size="sm" c="dimmed">
        {doneCount}/{comments.length} done
      </Text>
      <SegmentedControl
        size="xs"
        fullWidth
        value={filter}
        onChange={setFilter}
        data={[
          { label: "All", value: "all" },
          { label: "To do", value: "todo" },
          { label: "Done", value: "done" },
        ]}
      />
      <ScrollArea style={{ flex: 1 }}>
        <Stack gap={4}>
          {visible.map((c) => (
            <UnstyledButton
              key={c.comment_id}
              onClick={() => onSelect(c.comment_id)}
              px="xs"
              py={6}
              style={{
                borderRadius: 6,
                background:
                  c.comment_id === selectedId ? "var(--mantine-color-blue-light)" : "transparent",
              }}
            >
              <Group justify="space-between" wrap="nowrap">
                <Group gap={6} wrap="nowrap" style={{ overflow: "hidden" }}>
                  <Text size="xs" ff="monospace" c="dimmed">
                    {c.comment_id}
                  </Text>
                  <Badge size="xs" color={STRATUM_COLORS[c.stratum]} variant="light">
                    {c.topic_bucket || c.stratum}
                  </Badge>
                </Group>
                {c.done && (
                  <Text size="xs" c="teal" fw={700}>
                    ✓
                  </Text>
                )}
              </Group>
            </UnstyledButton>
          ))}
        </Stack>
      </ScrollArea>
    </Stack>
  );
}
