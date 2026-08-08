import { useEffect, useState } from "react";
import {
  Paper, Text, Stack, Group, Button, Badge, Switch, ActionIcon, ScrollArea,
  SimpleGrid,
} from "@mantine/core";
import { IconArrowLeft, IconArrowRight, IconPlus } from "@tabler/icons-react";
import { notifications } from "@mantine/notifications";
import { api } from "../api";
import PairEditor from "./PairEditor";
import ModelPanel from "./ModelPanel";
import ProposalPanel from "./ProposalPanel";
import CommentContext from "./CommentContext";

function emptyPair() {
  return {
    emotion: "", emotion_span: "", cause_span: "", cause_category: "",
    target_asset: "none", intensity: 0.5, sarcasm: false,
  };
}

function pairFromModel(mp, source) {
  return {
    emotion: mp.emotion || "",
    emotion_span: mp.emotion_span || "",
    cause_span: mp.cause_span ?? "",
    cause_category: mp.cause_category || "",
    target_asset: mp.target_asset || "none",
    intensity: typeof mp.intensity === "number" ? mp.intensity : 0.5,
    sarcasm: !!mp.sarcasm,
    // Bookkeeping only, never shown: records that this pair started as a model
    // suggestion rather than being authored from scratch, and who proposed it.
    // Lets the eval report state plainly how much of the set was model-assisted
    // instead of leaving readers to assume it was independent.
    _accepted_from: source || (mp.backers ? mp.backers.join(",") : "model"),
  };
}

export default function CommentDetail({ commentId, annotator, schema, comments, onSelect, onSaved }) {
  const [data, setData] = useState(null);
  const [pairs, setPairs] = useState([]);
  const [neutral, setNeutral] = useState(false);
  const [saving, setSaving] = useState(false);

  useEffect(() => {
    let cancelled = false;
    api.comment(commentId, annotator).then((d) => {
      if (cancelled) return;
      setData(d);
      setNeutral(d.myLabel?.neutral ?? false);
      setPairs(d.myLabel?.pairs?.length ? d.myLabel.pairs : []);
    });
    return () => {
      cancelled = true;
    };
  }, [commentId, annotator]);

  if (!data) return <Text c="dimmed">Loading…</Text>;

  // Empty in blind mode — the server withholds model/consensus predictions so
  // the human pass stays independent of what the LLMs said.
  const models = data.models || [];

  function updatePair(i, patch) {
    setPairs((prev) => prev.map((p, idx) => (idx === i ? { ...p, ...patch } : p)));
  }

  function addPair(prefill) {
    setNeutral(false);
    setPairs((prev) => [...prev, { ...emptyPair(), ...prefill }]);
  }

  function removePair(i) {
    setPairs((prev) => prev.filter((_, idx) => idx !== i));
  }

  function goNext() {
    const idx = comments.findIndex((c) => c.comment_id === commentId);
    const next = comments[idx + 1];
    if (next) onSelect(next.comment_id);
  }
  function goPrev() {
    const idx = comments.findIndex((c) => c.comment_id === commentId);
    const prev = comments[idx - 1];
    if (prev) onSelect(prev.comment_id);
  }

  async function save() {
    setSaving(true);
    try {
      const body = neutral
        ? { annotator, neutral: true, pairs: [] }
        : {
            annotator,
            neutral: false,
            pairs: pairs.map((p) => ({ ...p, cause_span: p.cause_span === "" ? null : p.cause_span })),
          };
      await api.saveLabel(commentId, body);
      notifications.show({ message: "Saved", color: "teal" });
      onSaved();
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
            <Text size="xs" c="dimmed">
              score {data.score} · {new Date(data.created_utc * 1000).toISOString().slice(0, 10)}
            </Text>
          </Group>

          <CommentContext data={data} />

          {models.length > 0 && (
            <SimpleGrid cols={{ base: 1, sm: Math.min(2, models.length) }} spacing="md">
              {models.map((m) => (
                <Paper key={m.model} withBorder p="md">
                  <ModelPanel model={m} onAccept={(p) => addPair(pairFromModel(p, m.model))} />
                </Paper>
              ))}
            </SimpleGrid>
          )}
        </Stack>
      </ScrollArea>

      <ScrollArea style={{ flex: 1, height: "100%" }} type="auto">
        <Stack gap="sm" pr="xs">
          {data.proposals && (
            <Paper withBorder p="md" bg="var(--mantine-color-teal-light)"
                   style={{ borderColor: "var(--mantine-color-teal-6)", borderWidth: 2 }}>
              <ProposalPanel
                proposals={data.proposals}
                onAccept={(p) => addPair(pairFromModel(p))}
                onAcceptAll={(ps) => {
                  setNeutral(false);
                  setPairs(ps.map((p) => ({ ...emptyPair(), ...pairFromModel(p) })));
                }}
                onNeutral={() => { setNeutral(true); setPairs([]); }}
              />
            </Paper>
          )}

          <Switch
            label="Neutral (no economically-caused emotion)"
            checked={neutral}
            onChange={(e) => {
              const v = e.currentTarget.checked;
              setNeutral(v);
              if (v) setPairs([]);
            }}
          />

          {!neutral && (
            <>
              {pairs.map((p, i) => (
                <PairEditor key={i} index={i} pair={p} schema={schema} onChange={updatePair} onRemove={removePair} />
              ))}
              <Button variant="light" leftSection={<IconPlus size={14} />} onClick={() => addPair()}>
                Add pair
              </Button>
            </>
          )}

          <Button size="md" onClick={save} loading={saving} mt="sm">
            Save &amp; next
          </Button>
        </Stack>
      </ScrollArea>
    </Group>
  );
}
