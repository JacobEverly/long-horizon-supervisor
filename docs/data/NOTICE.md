# Trace release notice

`public-agent-traces-v0.jsonl.gz` contains agent trajectories produced while
running tasks derived from
[alibabagroup/terminal-bench-pro](https://huggingface.co/datasets/alibabagroup/terminal-bench-pro),
which identifies its public dataset as Apache-2.0. The pinned source revision is
`b79dec3e67acd466f90127cde0d24516d9132e74`.

This project transformed the source tasks by executing several model routes in
the Terminus 2 agent harness, recording ATIF trajectories, attaching outcome
metadata, replacing credential-shaped strings and personal identifiers, and
compressing the records as JSONL. Protected verifier files are not included.

The archive also contains model-generated material whose applicable terms may
differ from the benchmark license. The combined release therefore uses
`license: other`; users are responsible for confirming that their intended use
complies with the relevant provider and source terms.
