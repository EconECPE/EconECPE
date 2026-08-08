import { useEffect, useState } from "react";
import {
  Text, Stack, Group, Button, Badge, ActionIcon, ScrollArea, SimpleGrid,
  SegmentedControl, Paper, Divider,
} from "@mantine/core";
import { IconArrowLeft, IconArrowRight, IconPlus } from "@tabler/icons-react";
import { notifications } from "@mantine/notifications";
import { api } from "../api";
import PairEditor from "./PairEditor";
import AnnotatorPanel from "./AnnotatorPanel";
import CommentContext from "./CommentContext";
import DiscussionThread from "./DiscussionThread";

function emptyPair() {
  return {
    emotion: "", emotion_span: "", cause_span: "", cause_category: "",
    target_asset: "none", intensity: 0.5, sarcasm: false,
  };
}

function pairFromExisting(p) {
  return {
    emotion: p.emotion || "",
    emotion_span: p.emotion_span || "",
    cause_span: p.cause_span ?? "",
    cause_category: p.cause_category || "",
    target_asset: p.target_asset || "none",
    intensity: typeof p.intensity === "number" ? p.intensity : 0.5,
    sarcasm: !!p.sarcasm,
  };
}

const FINAL_OPTIONS = [
  { label: "A", value: "A" },
  { label: "B", value: "B" },
  { label: "Neutral", value: "NEUTRAL" },
  { label: "Custom", value: "CUSTOM" },
];

export default function AdjudicationDetail({ commentId, annotator, schema, items, onSelect, onResolved }) {
  const [data, setData] = useState(null);
  const [final, setFinal] = useState(null);
  const [customPairs, setCustomPairs] = useState([]);
  const [saving, setSaving] = useState(false);

  useEffect(() => {
    let cancelled = false;
    api.adjudicationItem(commentId).then((d) => {
      if (cancelled) return;
      setData(d);
      setFinal(d.adjudication?.final ?? null);
      setCustomPairs(d.adjudication?.final === "CUSTOM" ? d.adjudication.pairs.map(pairFromExisting) : []);
    });
    return () => {
      cancelled = true;
    };
  }, [commentId]);

  if (!data) return <Text c="dimmed">Loading…</Text>;

  function useAsCustomPair(p) {
    setFinal("CUSTOM");
    setCustomPairs((prev) => [...prev, pairFromExisting(p)]);
  }

  function updatePair(i, patch) {
    setCustomPairs((prev) => prev.map((p, idx) => (idx === i ? { ...p, ...patch } : p)));
  }
  function removePair(i) {
    setCustomPairs((prev) => prev.filter((_, idx) => idx !== i));
  }

  function goNext() {
    const idx = items.findIndex((c) => c.comment_id === commentId);
    const next = items[idx + 1];
    if (next) onSelect(next.comment_id);
  }
  function goPrev() {
    const idx = items.findIndex((c) => c.comment_id === commentId);
    const prev = items[idx - 1];
    if (prev) onSelect(prev.comment_id);
  }

  async function save() {
    if (!final) {
      notifications.show({ title: "Pick a final call", message: "A, B, Neutral, or Custom", color: "red" });
      return;
    }
    setSaving(true);
    try {
      const body = { annotator, final };
      if (final === "CUSTOM") {
        body.pairs = customPairs.map((p) => ({ ...p, cause_span: p.cause_span === "" ? null : p.cause_span }));
      }
      await api.saveAdjudication(commentId, body);
      notifications.show({ message: "Saved as gold", color: "teal" });
      onResolved();
      goNext();
    } catch (e) {
      notifications.show({
        title: "Could not save",
        message: (e.details?.errors || [e.message]).join("\n"),
        color: "red",
        autoClose: 8000,
      });
    } finally {
      setSaving(false);
    }
  }

  return (
    <Group align="flex-start" wrap="nowrap" gap="md" h="calc(100vh - 92px)">
      <ScrollArea style={{ flex: 1.4, height: "100%" }} type="auto">
        <Stack gap="md" pr="md">
          <Group justify="space-between">
            <Group gap="xs">
              <ActionIcon variant="subtle" onClick={goPrev}>
                <IconArrowLeft size={16} />
              </ActionIcon>
              <Text size="sm" ff="monospace" c="dimmed">
                {data.comment_id}
              </Text>
              <Badge variant="light">{data.topic_bucket || data.stratum}</Badge>
              <ActionIcon variant="subtle" onClick={goNext}>
                <IconArrowRight size={16} />
              </ActionIcon>
            </Group>
            {data.adjudication && (
              <Text size="xs" c="dimmed">
                resolved {data.adjudication.final} by {data.adjudication.resolvedBy}
              </Text>
            )}
          </Group>

          <CommentContext data={data} />

          <SimpleGrid cols={2} spacing="md">
            <Paper withBorder p="md">
              <AnnotatorPanel label="A" record={data.a} onUsePair={useAsCustomPair} />
            </Paper>
            <Paper withBorder p="md">
              <AnnotatorPanel label="B" record={data.b} onUsePair={useAsCustomPair} />
            </Paper>
          </SimpleGrid>

          <Paper withBorder p="md" bg="var(--mantine-color-teal-light)"
                 style={{ borderColor: "var(--mantine-color-teal-6)", borderWidth: 2 }}>
            <AnnotatorPanel label="Silver consensus (reference only)" record={{
              neutral: data.consensus.pairs.length === 0, pairs: data.consensus.pairs,
            }} onUsePair={useAsCustomPair} />
          </Paper>
        </Stack>
      </ScrollArea>

      <ScrollArea style={{ flex: 1, height: "100%" }} type="auto">
        <Stack gap="sm" pr="xs">
          <Text size="xs" fw={700} c="dimmed" tt="uppercase">
            Final call
          </Text>
          <SegmentedControl fullWidth data={FINAL_OPTIONS} value={final || ""} onChange={setFinal} />

          {final === "CUSTOM" && (
            <>
              {customPairs.map((p, i) => (
                <PairEditor key={i} index={i} pair={p} schema={schema} onChange={updatePair} onRemove={removePair} />
              ))}
              <Button
                variant="light"
                leftSection={<IconPlus size={14} />}
                onClick={() => setCustomPairs((prev) => [...prev, emptyPair()])}
              >
                Add pair
              </Button>
            </>
          )}

          <Button size="md" onClick={save} loading={saving} mt="sm">
            Save as gold &amp; next
          </Button>

          <Divider my="xs" />

          <DiscussionThread commentId={commentId} annotator={annotator} initialMessages={data.messages} />
        </Stack>
      </ScrollArea>
    </Group>
  );
}
