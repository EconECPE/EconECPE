import { useState } from "react";
import { ScrollArea, Stack, UnstyledButton, Group, Text, Badge, SegmentedControl } from "@mantine/core";

export default function AdjudicationList({ items, selectedId, onSelect }) {
  const [filter, setFilter] = useState("open");
  const resolvedCount = items.filter((c) => c.resolved).length;

  const visible = items.filter((c) => {
    if (filter === "resolved") return c.resolved;
    if (filter === "open") return !c.resolved;
    return true;
  });

  return (
    <Stack h="100%" gap="xs">
      <Text size="sm" c="dimmed">
        {resolvedCount}/{items.length} resolved
      </Text>
      <SegmentedControl
        size="xs"
        fullWidth
        value={filter}
        onChange={setFilter}
        data={[
          { label: "Open", value: "open" },
          { label: "Resolved", value: "resolved" },
          { label: "All", value: "all" },
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
                  <Badge size="xs" variant="light">
                    {c.topic_bucket || c.stratum}
                  </Badge>
                </Group>
                {c.resolved ? (
                  <Badge size="xs" color="teal" variant="light">
                    {c.final}
                  </Badge>
                ) : (
                  <Text size="xs" c="orange" fw={700}>
                    ●
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
