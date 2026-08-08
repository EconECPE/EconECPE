import { useCallback, useEffect, useState } from "react";
import { AppShell, Group, Select, Title, Badge, Text, Loader, Center, SegmentedControl, Button } from "@mantine/core";
import { notifications } from "@mantine/notifications";
import { api } from "./api";
import CommentList from "./components/CommentList";
import CommentDetail from "./components/CommentDetail";
import AdjudicationList from "./components/AdjudicationList";
import AdjudicationDetail from "./components/AdjudicationDetail";

const ANNOTATOR_KEY = "gold-tool:annotator";

export default function App() {
  const [schema, setSchema] = useState(null);
  const [annotator, setAnnotator] = useState(localStorage.getItem(ANNOTATOR_KEY) || "");
  const [view, setView] = useState("label");
  const [comments, setComments] = useState([]);
  const [selectedId, setSelectedId] = useState(null);
  const [progress, setProgress] = useState(null);

  const [adjItems, setAdjItems] = useState([]);
  const [adjSelectedId, setAdjSelectedId] = useState(null);
  const [adjProgress, setAdjProgress] = useState(null);
  const [exporting, setExporting] = useState(false);

  useEffect(() => {
    api.schema().then((s) => {
      setSchema(s);
      // A name remembered from another run (e.g. "A" from the 300-comment
      // pass) isn't valid against a different roster — drop it and re-ask,
      // rather than firing requests the server will reject.
      setAnnotator((prev) => {
        if (prev && !s.annotators.includes(prev)) {
          localStorage.removeItem(ANNOTATOR_KEY);
          return "";
        }
        return prev;
      });
      if (!s.adjudication) setView("label");
    });
  }, []);

  const refreshList = useCallback(async (ann) => {
    if (!ann) return;
    const [list, prog] = await Promise.all([api.comments(ann), api.progress()]);
    setComments(list);
    setProgress(prog);
    setSelectedId((prev) => prev ?? list[0]?.comment_id ?? null);
  }, []);

  const refreshAdjudication = useCallback(async () => {
    const [list, prog] = await Promise.all([api.adjudicationList(), api.adjudicationProgress()]);
    setAdjItems(list);
    setAdjProgress(prog);
    setAdjSelectedId((prev) => prev ?? list.find((c) => !c.resolved)?.comment_id ?? list[0]?.comment_id ?? null);
  }, []);

  useEffect(() => {
    refreshList(annotator);
  }, [annotator, refreshList]);

  useEffect(() => {
    if (view === "adjudicate") refreshAdjudication();
  }, [view, refreshAdjudication]);

  function chooseAnnotator(value) {
    if (!value) return;
    localStorage.setItem(ANNOTATOR_KEY, value);
    setAnnotator(value);
  }

  async function exportGold() {
    setExporting(true);
    try {
      const res = await api.exportGold();
      notifications.show({
        title: "Gold standard exported",
        message: `Wrote ${res.n} comments to ${res.path}`,
        color: "teal",
      });
    } catch (e) {
      notifications.show({
        title: "Not ready to export",
        message: e.message,
        color: "red",
        autoClose: 8000,
      });
    } finally {
      setExporting(false);
    }
  }

  if (!schema) {
    return (
      <Center h="100vh">
        <Loader />
      </Center>
    );
  }

  if (!annotator) {
    return (
      <Center h="100vh">
        <Select
          label="Who's labeling?"
          placeholder="Pick your name"
          data={schema.annotators}
          onChange={chooseAnnotator}
          w={280}
          searchable={false}
          allowDeselect={false}
        />
      </Center>
    );
  }

  const doneCount = comments.filter((c) => c.done).length;

  return (
    <AppShell navbar={{ width: 320, breakpoint: "sm" }} header={{ height: 60 }} padding="md">
      <AppShell.Header>
        <Group h="100%" px="md" justify="space-between" wrap="nowrap">
          <Group gap="md" wrap="nowrap">
            <Title order={4}>
              {schema.blind ? "EconECPE human gold pass" : "EconECPE gold-label review"}
            </Title>
            {schema.blind && (
              <Badge variant="light" color="grape" size="lg">
                blind · model labels hidden
              </Badge>
            )}
            {schema.adjudication && (
              <SegmentedControl
                size="xs"
                value={view}
                onChange={setView}
                data={[
                  { label: "My labels", value: "label" },
                  { label: "Adjudication", value: "adjudicate" },
                ]}
              />
            )}
          </Group>
          <Group gap="xs" wrap="nowrap">
            {view === "label" &&
              progress &&
              schema.annotators.map((a) => (
                <Badge key={a} variant={a === annotator ? "filled" : "light"} size="lg">
                  {a}: {progress.counts[a] ?? 0}/{progress.total}
                </Badge>
              ))}
            {view === "adjudicate" && adjProgress && (
              <>
                <Badge size="lg" variant="light" color={adjProgress.resolved === adjProgress.total ? "teal" : "orange"}>
                  {adjProgress.resolved}/{adjProgress.total} resolved
                </Badge>
                <Button
                  size="xs"
                  variant="light"
                  loading={exporting}
                  disabled={adjProgress.resolved < adjProgress.total}
                  onClick={exportGold}
                >
                  Export gold standard
                </Button>
              </>
            )}
            <Select
              w={130}
              data={schema.annotators}
              value={annotator}
              onChange={chooseAnnotator}
              allowDeselect={false}
            />
          </Group>
        </Group>
      </AppShell.Header>
      <AppShell.Navbar p="sm">
        {view === "label" ? (
          <CommentList
            comments={comments}
            selectedId={selectedId}
            onSelect={setSelectedId}
            doneCount={doneCount}
          />
        ) : (
          <AdjudicationList items={adjItems} selectedId={adjSelectedId} onSelect={setAdjSelectedId} />
        )}
      </AppShell.Navbar>
      <AppShell.Main>
        {view === "label" ? (
          selectedId ? (
            <CommentDetail
              key={`${selectedId}:${annotator}`}
              commentId={selectedId}
              annotator={annotator}
              schema={schema}
              comments={comments}
              onSelect={setSelectedId}
              onSaved={() => refreshList(annotator)}
            />
          ) : (
            <Text c="dimmed">No comments loaded.</Text>
          )
        ) : adjSelectedId ? (
          <AdjudicationDetail
            key={`${adjSelectedId}:${annotator}`}
            commentId={adjSelectedId}
            annotator={annotator}
            schema={schema}
            items={adjItems}
            onSelect={setAdjSelectedId}
            onResolved={refreshAdjudication}
          />
        ) : (
          <Text c="dimmed">No disagreements to adjudicate.</Text>
        )}
      </AppShell.Main>
    </AppShell>
  );
}
