import { sameQuality, summarize } from "./inspector-data.js";

// JSON object key order is irrelevant when checking repeated evidence.
const canonical = (value) => JSON.stringify(value, (_, item) =>
  item && typeof item === "object" && !Array.isArray(item)
    ? Object.fromEntries(Object.keys(item).sort().map((key) => [key, item[key]])) : item);
const same = (left, right) => canonical(left) === canonical(right);

function policyKey({ mode, standard, description, completeness, contamination }) {
  if (mode === "thresholds") return canonical([mode, completeness, contamination]);
  return canonical([mode, standard, standard ? null : description]);
}

function makeIds(datasets) {
  const owners = new Map();
  const keys = new Map();
  return {
    add(dataset, id) {
      if (!owners.has(id)) owners.set(id, new Set());
      owners.get(id).add(dataset);
    },
    get(dataset, id) {
      if (owners.get(id).size === 1) return id;
      const key = JSON.stringify([dataset, id]);
      if (!keys.has(key)) {
        let label = `[${datasets.get(dataset)}] ${id}`;
        while (owners.has(label)) label = `_${label}`;
        keys.set(key, label);
        owners.set(label, new Set([dataset]));
      }
      return keys.get(key);
    },
  };
}

function combineBin(previous, row, name, datasetNumber) {
  const old = previous.row;
  const conflict = (field) => {
    throw new Error(`Dataset ${datasetNumber}, bin ${row.id} has conflicting ${field} in ${previous.name} and ${name}.`);
  };
  if (!sameQuality(old.original, row.original)) conflict("original measurements");
  if (old.selected_proposal_id && row.selected_proposal_id && old.selected_proposal_id !== row.selected_proposal_id) {
    conflict("selected proposals");
  }
  if (old.selected_proposal_id === row.selected_proposal_id
      && (!sameQuality(old.final, row.final) || old.hq_transition !== row.hq_transition || !same(old.delta, row.delta)
        || !same([...(old.assessment_mismatches ?? [])].sort(), [...(row.assessment_mismatches ?? [])].sort()))) {
    conflict("final evidence");
  }
  if (old.contig_evidence && row.contig_evidence && !same(old.contig_evidence, row.contig_evidence)) conflict("contig evidence");
  for (const proposal of row.proposals) {
    const existing = previous.proposals.get(proposal.id);
    if (existing && !same(existing.proposal, proposal)) {
      throw new Error(`Proposal ${proposal.id} conflicts in ${existing.name} and ${name}.`);
    }
    if (!existing) previous.proposals.set(proposal.id, { proposal, name });
  }
  const selected = row.selected_proposal_id && !old.selected_proposal_id;
  const evidence = row.contig_evidence ? row : old;
  previous.row = {
    ...(selected ? row : old), recipient: old.recipient || row.recipient,
    contig_evidence: evidence.contig_evidence, contigRnaSource: evidence.contigRnaSource,
  };
  if (selected) previous.name = name;
}

function sharedMethodFields(rows) {
  const fields = new Map();
  const conflicts = new Set();
  for (const row of rows) {
    for (const [field, value] of Object.entries(row)) {
      if (!fields.has(field)) fields.set(field, value);
      else if (!same(fields.get(field), value)) conflicts.add(field);
    }
  }
  return Object.fromEntries([...fields].filter(([field]) => !conflicts.has(field)));
}

export function mergeReports(inputs) {
  const first = inputs[0];
  const policy = policyKey(first.data.hq_policy);
  const datasets = new Map();
  const binIds = makeIds(datasets);
  const proposalIds = makeIds(datasets);
  const artifactIds = makeIds(datasets);
  const datasetBins = new Map();
  const datasetProposals = new Map();
  const artifactMethods = new Map();
  function addArtifact(dataset, id, method, name) {
    const key = JSON.stringify([dataset, id]);
    const existing = artifactMethods.get(key);
    if (existing && existing.method !== method) {
      throw new Error(`Artifact ${id} has different method names in ${existing.name} and ${name}.`);
    }
    artifactMethods.set(key, { method, name });
    artifactIds.add(dataset, id);
  }
  for (const { name, data } of inputs) {
    if (policyKey(data.hq_policy) !== policy) {
      throw new Error(`${name} and ${first.name} use different HQ policies.`);
    }
    const dataset = data.provenance.dataset_identity;
    if (!datasets.has(dataset)) {
      datasets.set(dataset, datasets.size + 1);
      datasetBins.set(dataset, new Map());
      datasetProposals.set(dataset, new Map());
    }
    const rows = datasetBins.get(dataset);
    const proposalOwners = datasetProposals.get(dataset);
    for (const [id, { method }] of Object.entries(data.comparison?.methods ?? {})) addArtifact(dataset, id, method, name);
    for (const bin of data.bins) {
      binIds.add(dataset, bin.id);
      for (const proposal of bin.proposals) {
        const owner = proposalOwners.get(proposal.id);
        if (owner && owner.id !== bin.id) {
          throw new Error(`Proposal ${proposal.id} belongs to different bins in ${owner.name} and ${name}.`);
        }
        proposalOwners.set(proposal.id, { id: bin.id, name });
        proposalIds.add(dataset, proposal.id);
        addArtifact(dataset, proposal.artifact_id, proposal.method, name);
      }
      const row = { ...bin, uncheckedCheckm2Settings: Boolean(data.uncheckedCheckm2Settings),
        contigRnaSource: data.provenance.contig_rna_source };
      if (rows.has(bin.id)) combineBin(rows.get(bin.id), row, name, datasets.get(dataset));
      else rows.set(bin.id, { name, row, proposals: new Map(bin.proposals.map((proposal) => [proposal.id, { proposal, name }])) });
    }
  }

  const bins = [];
  const byBin = new Map();
  const byProposal = new Map();
  for (const [dataset, rows] of datasetBins) {
    const selectedDonors = new Map();
    for (const { row, proposals, name } of rows.values()) {
      const selected = proposals.get(row.selected_proposal_id)?.proposal;
      for (const donor of selected?.donors ?? []) {
        const existing = selectedDonors.get(donor.id);
        if (existing) {
          throw new Error(`Selected bins ${existing.id} and ${row.id} share contig ${donor.id} in ${existing.name} and ${name}.`);
        }
        selectedDonors.set(donor.id, { id: row.id, name });
      }
      const id = binIds.get(dataset, row.id);
      const merged = { ...row, id,
        selected_proposal_id: row.selected_proposal_id === null ? null : proposalIds.get(dataset, row.selected_proposal_id),
        proposals: [...proposals.values()].map(({ proposal }) => ({ ...proposal,
          id: proposalIds.get(dataset, proposal.id), artifact_id: artifactIds.get(dataset, proposal.artifact_id),
        })),
      };
      bins.push(merged);
      byBin.set(id, merged);
      for (const proposal of merged.proposals) byProposal.set(proposal.id, { ...proposal, recipient: id });
    }
  }

  const savedMethods = new Map();
  const sourceReports = inputs.map(({ name, data }) => {
    const dataset = data.provenance.dataset_identity;
    for (const [id, row] of Object.entries(data.comparison?.methods ?? {})) {
      const key = artifactIds.get(dataset, id);
      if (!savedMethods.has(key)) savedMethods.set(key, []);
      savedMethods.get(key).push(row);
    }
    const { comparison, provenance, summary, hq_policy, schema_version } = data;
    return { name, datasetNumber: datasets.get(dataset), comparison, provenance, summary, hq_policy, schema_version };
  });
  const methodArtifacts = Object.fromEntries([...savedMethods].map(([key, rows]) => [key, sharedMethodFields(rows)]));
  const combined = { bins, hq_policy: first.data.hq_policy };
  const totals = summarize(combined);
  return { ...combined, summary: totals.summary,
    rna_summary: inputs.some(({ data }) => data.rna_summary) ? totals.rna_summary : undefined,
    synthetic: inputs.every(({ data }) => data.synthetic), sourceReports, methodArtifacts,
    uncheckedCheckm2Settings: inputs.some(({ data }) => data.uncheckedCheckm2Settings),
    byBin, byProposal, proposals: [...byProposal.values()],
  };
}
