import { useEffect, useRef, useState } from "react";
import { ScrollArea, Stack, Group, Text, Textarea, Button, Paper } from "@mantine/core";
import { api } from "../api";

const POLL_MS = 4000;

// Lightweight threaded chat per disagreement, keyed by comment_id. Polls
// instead of using a websocket since this is a two-person internal tool
// (gold-tool/README: LAN or same-host usage) — good enough to feel live.
export default function DiscussionThread({ commentId, annotator, initialMessages }) {
  const [messages, setMessages] = useState(initialMessages || []);
  const [draft, setDraft] = useState("");
  const [sending, setSending] = useState(false);
  const viewportRef = useRef(null);

  useEffect(() => {
    setMessages(initialMessages || []);
  }, [commentId, initialMessages]);

  useEffect(() => {
    const id = setInterval(() => {
      api.messages(commentId).then(setMessages).catch(() => {});
    }, POLL_MS);
    return () => clearInterval(id);
  }, [commentId]);

  useEffect(() => {
    const el = viewportRef.current;
    if (el) el.scrollTop = el.scrollHeight;
  }, [messages]);

  async function send() {
    const body = draft.trim();
    if (!body) return;
    setSending(true);
    try {
      const next = await api.postMessage(commentId, annotator, body);
      setMessages(next);
      setDraft("");
    } finally {
      setSending(false);
    }
  }

  return (
    <Stack gap="xs">
      <Text size="xs" fw={700} c="dimmed" tt="uppercase">
        Discussion
      </Text>
      <ScrollArea.Autosize mah={220} viewportRef={viewportRef} type="auto">
        <Stack gap={6}>
          {messages.length === 0 && (
            <Text size="sm" c="dimmed">
              No messages yet — say why you think it should go one way or the other.
            </Text>
          )}
          {messages.map((m, i) => (
            <Paper
              key={i}
              withBorder
              p={6}
              radius="md"
              bg={m.author === annotator ? "var(--mantine-color-blue-light)" : undefined}
            >
              <Group gap={6} justify="space-between">
                <Text size="xs" fw={700}>
                  {m.author}
                </Text>
                <Text size="xs" c="dimmed">
                  {new Date(m.created_at).toLocaleString()}
                </Text>
              </Group>
              <Text size="sm" style={{ whiteSpace: "pre-wrap" }}>
                {m.body}
              </Text>
            </Paper>
          ))}
        </Stack>
      </ScrollArea.Autosize>
      <Group gap="xs" wrap="nowrap" align="flex-end">
        <Textarea
          style={{ flex: 1 }}
          placeholder={`Message as ${annotator}…`}
          value={draft}
          onChange={(e) => setDraft(e.currentTarget.value)}
          autosize
          minRows={1}
          maxRows={4}
          onKeyDown={(e) => {
            if (e.key === "Enter" && !e.shiftKey) {
              e.preventDefault();
              send();
            }
          }}
        />
        <Button onClick={send} loading={sending} disabled={!draft.trim()}>
          Send
        </Button>
      </Group>
    </Stack>
  );
}
