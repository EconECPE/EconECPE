import { Paper, Text, Divider } from "@mantine/core";

// Post title/body, parent comment, and the comment itself — the same context
// shown to both an annotator labeling a comment and to A/B
// adjudicating a disagreement on it.
export default function CommentContext({ data }) {
  return (
    <Paper withBorder p="md">
      {data.post_title && (
        <>
          <Text size="xs" fw={700} c="dimmed" tt="uppercase" mb={4}>
            Post title
          </Text>
          <Text size="sm" mb="sm" style={{ whiteSpace: "pre-wrap" }}>
            {data.post_title}
          </Text>
        </>
      )}
      {data.post_selftext && (
        <Text size="sm" mb="sm" style={{ whiteSpace: "pre-wrap" }}>
          {data.post_selftext}
        </Text>
      )}
      {data.parent_body && (
        <>
          <Divider my="sm" label="parent comment" labelPosition="left" />
          <Text size="sm" c="dimmed" style={{ whiteSpace: "pre-wrap" }}>
            {data.parent_body}
          </Text>
        </>
      )}
      <Divider my="sm" label={`comment (u/${data.author})`} labelPosition="left" />
      <Text size="sm" fw={500} style={{ whiteSpace: "pre-wrap", lineHeight: 1.6 }}>
        {data.body}
      </Text>
    </Paper>
  );
}
