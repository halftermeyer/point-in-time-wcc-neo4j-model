# Point-in-Time WCC Model

*Real-time component retrieval in constant time per member, for temporal graphs in Neo4j*

---

## The Problem: Future Leakage in Graph-Based Feature Engineering

Weakly Connected Components (WCC) are among the most powerful features for fraud detection in machine learning. When entities share identifiers across events — a phone number reused across accounts, a device appearing in multiple transactions — WCC captures the hidden structure that individual records cannot reveal. Component size, growth velocity, and diameter become strong predictive signals.

But there is a fundamental problem: **WCC is not a point-in-time operation.**

Running WCC on the current state of a graph produces the component as it exists *today* — not as it existed when a given event occurred. When this result is used as a feature for historical records during model training, future information leaks into the past. Events that hadn't happened yet at prediction time silently influence the training data.

This is **future leakage**, and it invalidates the model.

Consider a graph where event `e₁` occurs at time `t₁` and belongs to a small component of 3 events. By time `t₂`, that component has merged with another and now contains 50 events. If we compute WCC at `t₂` and assign `component_size = 50` to `e₁`, we have leaked the knowledge of 47 events that occurred *after* `e₁`. The model learns to exploit information it will never have at inference time, producing inflated metrics that collapse in production.

The Point-in-Time WCC Model solves this by embedding the full temporal history of component evolution directly into the graph structure, enabling:

- **Component retrieval as of any event's own timestamp** — for leak-free ML training data
- **Component retrieval as of any arbitrary date** — for historical analysis
- **Component retrieval as of now** — for real-time inference
- **Sub-millisecond query performance** — via pre-materialized traversal chains, no WCC recomputation required


## Architecture: Training vs. Inference

The Point-in-Time WCC Model serves two distinct operational paths that share the same data model but run on different infrastructure.

### Training Path

Raw data is loaded into an **analytics Neo4j instance** equipped with the Graph Data Science (GDS) library. The batch build pipeline constructs the full temporal component structure from an existing Event–Thing graph. Point-in-time queries then extract component features for each event **as of its own timestamp**, producing a leak-free training dataset. This enriched data feeds the ML model training process.

### Inference Path

In production, new events arrive at an **HA Neo4j transactional cluster**. Each event is ingested via a single Cypher transaction that atomically updates the component structure — creating new components, extending existing ones, or merging multiple components. Features are extracted immediately from the updated structure and passed to the trained ML model for real-time scoring.

The same data model and the same query patterns work identically across both paths.


## Foundation: The Event–Thing Bipartite Graph

The model is built on a simple, domain-agnostic bipartite structure:

```
(:Event {event_id, timestamp})-[:WITH]->(:Thing {thing_id})
```

An **Event** is any timestamped occurrence: a transaction, a login, a border crossing, a claim submission. A **Thing** is any identifier or entity observed during that event: a phone number, an account, a device fingerprint, a document number.

Events connect to Things. Two events that share a Thing are implicitly connected. The union of all such implicit connections defines the weakly connected components of the graph — clusters of events linked by shared identifiers across time.

The Thing node carries domain-specific labels as subtypes. In a fraud detection context, Thing subtypes might include `:Phone`, `:Device`, `:Account`, `:Email`. In a border security context: `:TravelDocument`, `:BiometricData`, `:FlightBooking`. The model is agnostic to these subtypes — it operates purely on the Event–Thing bipartite topology.


## The Temporal Component Structure

Three relationship types and one auxiliary label are overlaid on the Event–Thing foundation to enable point-in-time component retrieval.

### Node Labels

| Label | Description |
|---|---|
| `Event` | Every event in the graph. Carries `event_id` and `timestamp`. |
| `Thing` | Every identifier/entity observed during events. Carries `thing_id`. |
| `ComponentNode` | Added to every Event that participates in the temporal component structure. Every Event becomes a ComponentNode once processed. |

### Relationship Types

| Relationship | Direction | Description |
|---|---|---|
| `WITH` | `Event → Thing` | The base bipartite link. An event observed a thing. |
| `COMPONENT_PARENT` | `ComponentNode → ComponentNode` | Points forward in time from an older component head to the newer component head that supersedes it. Forms a forest with out-degree ≤ 1 and in-degree ≥ 0. |
| `DFS_NEXT` | `ComponentNode → ComponentNode` | A pre-materialized DFS traversal order within a component. Forms a linked list from the component head through all its members. |
| `LAST_DFS_NODE_IN_COMP` | `ComponentNode → ComponentNode` | Points from a component head to the last node in its DFS chain. Enables bounded traversal without scanning to the end. |

### Structural Invariants

The `COMPONENT_PARENT` graph is a **chronologically oriented forest**:
- **Out-degree is 0 or 1.** A component head either has no successor (it is the current head) or exactly one successor (a later event subsumed it).
- **In-degree can be > 1.** When a new event merges N previously separate components, all N old heads point to the new event.
- **Roots are the current component heads.** They have no outgoing `COMPONENT_PARENT`.

The `DFS_NEXT` chain for each component head forms a **linked list** starting at the head and ending at the node referenced by `LAST_DFS_NODE_IN_COMP`. This chain contains every ComponentNode that belongs to that component at that point in time.

### How Components Evolve

When a new event `e` arrives, three cases are possible:

**Case 1: No existing component matched.** The event's Things are all new, or none belong to any existing component. The event becomes a standalone component — `ComponentNode` with `LAST_DFS_NODE_IN_COMP` pointing to itself.

**Case 2: One existing component matched.** The event's Things overlap with exactly one existing component whose current head is `y`. The structure updates as:
- `y -[:COMPONENT_PARENT]-> e` — the old head points forward to the new head
- `e -[:DFS_NEXT]-> y` — the new head's traversal starts with the old component
- `e -[:LAST_DFS_NODE_IN_COMP]-> z` — where `z` was the old tail (the previous target of `y`'s `LAST_DFS_NODE_IN_COMP`)

**Case 3: Multiple existing components matched (merge).** The event's Things span N ≥ 2 components with current heads `h₁, h₂, ..., hₙ` (ordered by `event_id`). The structure updates as:
- All old heads get `COMPONENT_PARENT` pointing to `e`
- Their DFS chains are **concatenated**: the tail of `h₁` gets `DFS_NEXT` to `h₂`, the tail of `h₂` gets `DFS_NEXT` to `h₃`, and so on
- The new event is prepended: `e -[:DFS_NEXT]-> h₁`
- The new tail is the old tail of `hₙ`: `e -[:LAST_DFS_NODE_IN_COMP]-> tail(hₙ)`

In all cases, old component heads remain intact. They are never deleted or modified beyond the addition of a `COMPONENT_PARENT` edge. This is what makes point-in-time queries possible — every historical component state is preserved.


## Querying the Structure

All three query patterns exploit the same principle: the `DFS_NEXT` chain is a pre-materialized linked list. Retrieving a component is a simple chain walk — no WCC recomputation, no graph-wide traversal.

### Component as of Event Time

Retrieve the component exactly as it existed when a given event occurred:

```cypher
// Component as of event's own timestamp
MATCH path = (e:Event {event_id: $event_id})
  (()-[:DFS_NEXT]->(evs))*
  (last)<-[:LAST_DFS_NODE_IN_COMP]-(e)
UNWIND [e] + evs AS ev
RETURN ev
```

This walks the DFS chain starting at the event and terminates at the node marked by `LAST_DFS_NODE_IN_COMP`. The result is every event that was in the same component at the moment `$event_id` was recorded. This is the query used for **leak-free ML feature engineering**.

### Component as of Now (Latest State)

Retrieve the current state of the component that a given event belongs to:

```cypher
// Component as of today
MATCH fast_fw_to_future = (e:Event {event_id: $event_id})
  -[:COMPONENT_PARENT]->*
  (latest_future_comp:Event
    WHERE NOT EXISTS {
      (latest_future_comp)-[:COMPONENT_PARENT]->()
    }
  ),
  comp = (latest_future_comp)-[:DFS_NEXT]->*(last),
  (last)<-[:LAST_DFS_NODE_IN_COMP]-(latest_future_comp)
UNWIND nodes(comp) AS ev
RETURN ev
```

This first walks the `COMPONENT_PARENT` chain from the event to the current root (the head with no outgoing `COMPONENT_PARENT`), then walks that root's `DFS_NEXT` chain. The fast-forward is a simple linear walk because `COMPONENT_PARENT` out-degree is at most 1.

### Component as of Arbitrary Date

Retrieve the component as it existed at a specific point in time:

```cypher
// Component as of $asOfDate
MATCH fast_fw_to_future = (e:Event {event_id: $event_id}
    WHERE e.timestamp <= $asOfDate)
  (()-[:COMPONENT_PARENT]->(future WHERE future.timestamp <= $asOfDate))*
  (latest_future_comp:Event
    WHERE NOT EXISTS {
      (latest_future_comp)-[:COMPONENT_PARENT]->(x
        WHERE x.timestamp <= $asOfDate)
    }
  ),
  comp = (latest_future_comp)-[:DFS_NEXT]->*(last),
  (last)<-[:LAST_DFS_NODE_IN_COMP]-(latest_future_comp)
UNWIND nodes(comp) AS ev
RETURN ev
```

This applies a timestamp filter to the `COMPONENT_PARENT` walk, stopping at the latest head that existed on or before `$asOfDate`. This enables historical analysis at any granularity.

### Performance

Because the DFS chain is pre-materialized, component retrieval is a linked-list traversal with **constant time per member**. There is no WCC recomputation, no GDS call, and no traversal through Thing nodes. Typical query times are in the **low milliseconds** even for components with dozens of members.


## Building the Structure: Batch Pipeline

The batch pipeline constructs the full temporal component structure from an existing Event–Thing graph. It requires the Neo4j Graph Data Science (GDS) library and proceeds in four steps.

### Prerequisites

```cypher
CREATE CONSTRAINT event_id_unique IF NOT EXISTS
  FOR (e:Event) REQUIRE (e.event_id) IS UNIQUE;
CREATE CONSTRAINT thing_id_unique IF NOT EXISTS
  FOR (t:Thing) REQUIRE (t.thing_id) IS UNIQUE;
CREATE INDEX event_timestamp IF NOT EXISTS
  FOR (e:Event) ON (e.timestamp);
```

### Step 1: Build SEQUENTIALLY_RELATED

For each Thing, time-order the Events connected to it and chain consecutive pairs. This transforms implicit bipartite connectivity into explicit Event-to-Event edges.

```cypher
-- Project the bipartite graph
MATCH (source:Event)
OPTIONAL MATCH (source)-[:WITH]->(target)
RETURN gds.graph.project('event_thing_graph', source, target, {});
```

```cypher
-- Create SEQUENTIALLY_RELATED via WCC-batched processing
CALL gds.wcc.stream('event_thing_graph')
YIELD nodeId, componentId
WITH gds.util.asNode(nodeId) AS thing, componentId AS community
WHERE thing:Thing
WITH community, collect(thing) AS things
CALL (things) {
  UNWIND things AS thing
  CALL (thing) {
    MATCH (e:Event)-[:WITH]->(thing)
    WITH DISTINCT e ORDER BY e.timestamp
    WITH collect(e) AS events
    UNWIND range(0, size(events)-2) AS ix
    WITH events[ix] AS source, events[ix+1] AS target
    MERGE (source)-[:SEQUENTIALLY_RELATED]->(target)
  }
} IN 8 CONCURRENT TRANSACTIONS OF 100 ROWS
```

The WCC call is used here to batch processing by component, ensuring that SEQUENTIALLY_RELATED is created within connected sub-graphs for efficiency.

After this step, the SEQUENTIALLY_RELATED graph between Event nodes encodes exactly the same WCC structure as the original bipartite Event–Thing graph.

> **Note:** `SEQUENTIALLY_RELATED` is intermediate scaffolding. It can be dropped after the batch build unless incremental batch updates are planned.

### Step 2: Build COMPONENT_PARENT Forest

Project the SEQUENTIALLY_RELATED graph and process events chronologically within each WCC to build the union-find forest:

```cypher
-- Project the event-only graph
MATCH (source:Event)
OPTIONAL MATCH (source)-[:SEQUENTIALLY_RELATED]->(target)
RETURN gds.graph.project('seq_rel_event_graph', source, target, {});
```

```cypher
-- Build the COMPONENT_PARENT forest
CALL gds.wcc.stream('seq_rel_event_graph')
YIELD nodeId, componentId
WITH gds.util.asNode(nodeId) AS event, componentId
WITH componentId, collect(event) AS events
ORDER BY rand()  // distribute components across concurrent transactions
CALL (events) {
  UNWIND events AS e
  WITH e WHERE NOT e:ComponentNode
  ORDER BY e.timestamp ASC
  CALL (e) {
    SET e:ComponentNode
    WITH e
    MATCH (x:ComponentNode)-[:SEQUENTIALLY_RELATED]->(e)
    MATCH (x)-[:COMPONENT_PARENT]->*(cc
      WHERE NOT EXISTS {(cc)-[:COMPONENT_PARENT]->()})
    MERGE (cc)-[:COMPONENT_PARENT]->(e)
  }
} IN 8 CONCURRENT TRANSACTIONS OF 100 ROWS
```

This replays the online algorithm in batch: each event is processed in timestamp order, finds its neighbors' current component heads, and becomes the new head. The `ORDER BY rand()` distributes independent components across concurrent transactions for parallelism.

### Step 3: Build DFS_NEXT Chains

Project the COMPONENT_PARENT forest (reversed) and run GDS DFS from each root to produce traversal order:

```cypher
-- Project the component forest (reversed direction for DFS from roots)
MATCH (source:Event)
OPTIONAL MATCH (source)<-[:COMPONENT_PARENT]-(target)
RETURN gds.graph.project('component_forest', source, target, {});
```

```cypher
-- Create DFS_NEXT from root component heads
MATCH (source:ComponentNode)
WHERE NOT EXISTS {(source)-[:COMPONENT_PARENT]->()}
  AND EXISTS {(source)<-[:COMPONENT_PARENT]-()}
CALL (source) {
  CALL gds.dfs.stream('component_forest', {
    sourceNode: source
  })
  YIELD path
  WITH relationships(path) AS rels
  UNWIND rels AS rel
  WITH startNode(rel) AS n1, endNode(rel) AS n2
  MERGE (n1)-[:DFS_NEXT]->(n2)
} IN 8 CONCURRENT TRANSACTIONS OF 50 ROWS
```

### Step 4: Build LAST_DFS_NODE_IN_COMP Markers

Create the boundary markers that enable bounded traversal of each component's DFS chain:

```cypher
CYPHER 25
CALL () {
    -- Case 1: ComponentNode with a DFS_NEXT successor — find the last node
    -- in its own sub-tree (nodes reachable via COMPONENT_PARENT from it,
    -- but not from its DFS_NEXT successor)
    MATCH (c1:ComponentNode)-[:DFS_NEXT]->(c2:ComponentNode)
    MATCH (c1)(()-[:COMPONENT_PARENT]->(ps)
      WHERE NOT EXISTS {(c2)-[:COMPONENT_PARENT]->*(ps)}
    )*
    UNWIND ps AS p
    RETURN c1, p

  UNION

    -- Case 2: ComponentNode with no DFS_NEXT successor — it is a leaf,
    -- walk its full COMPONENT_PARENT sub-tree
    MATCH (c1:ComponentNode WHERE NOT EXISTS {(c1)-[:DFS_NEXT]->()})
    MATCH (c1)(()-[:COMPONENT_PARENT]->(ps))*
    UNWIND ps AS p
    RETURN c1, p

  UNION

    -- Case 3: ComponentNode with no incoming COMPONENT_PARENT —
    -- standalone single-event component, points to itself
    MATCH (c1:ComponentNode WHERE NOT EXISTS {()-[:COMPONENT_PARENT]->(c1)})
    RETURN c1, c1 AS p

}
CALL (c1, p) {
  MERGE (p)-[:LAST_DFS_NODE_IN_COMP]->(c1)
} IN TRANSACTIONS OF 100 ROWS
```


## Building the Structure: Online (Single Event Ingestion)

When a new event arrives in the transactional system, a single Cypher query atomically updates the component structure. This requires no GDS library — it runs on pure Cypher.

```cypher
CYPHER 25
LET event = {
  event_id: $event_id,
  things: $things
}
WITH event, collect {
  UNWIND event.things AS th
  MATCH (t:Thing {thing_id: th.thing_id})<-[:WITH]-(c:ComponentNode)
  RETURN c
} AS matched_comps
CALL (event, matched_comps) {
  // Case 1: No existing components matched — create standalone component
  WHEN size(matched_comps) = 0 THEN {
    CREATE (e:Event {event_id: event.event_id})
    SET e.timestamp = datetime(),
        e:ComponentNode
    UNWIND event.things AS th
    MERGE (t:Thing {thing_id: th.thing_id})
    ON CREATE SET t:$(th.labels)
    MERGE (e)-[:WITH]->(t)
    MERGE (e)-[:LAST_DFS_NODE_IN_COMP]->(e)
    RETURN e
  }
  // Case 2+3: One or more existing components matched — extend or merge
  ELSE {
    CREATE (e:Event {event_id: event.event_id})
    SET e.timestamp = datetime(),
        e:ComponentNode
    UNWIND event.things AS th
    MERGE (t:Thing {thing_id: th.thing_id})
    ON CREATE SET t:$(th.labels)
    MERGE (e)-[:WITH]->(t)
    // Walk each Thing's ComponentNode up to its current root
    CALL (t) {
      MATCH (t)<-[:WITH]-(ev:ComponentNode)
      LIMIT 1
      MATCH (ev)-[:COMPONENT_PARENT]->*(parent
        WHERE NOT EXISTS {(parent)-[:COMPONENT_PARENT]->()}
      )
      RETURN parent
    }
    WITH DISTINCT event, e, parent
    ORDER BY parent.event_id
    WITH event, e, collect(parent) AS sub_comps
    // Stitch DFS chains of sub-components together
    CALL (sub_comps) {
      UNWIND range(0, size(sub_comps)-2) AS ix
      WITH sub_comps[ix] AS comp1, sub_comps[ix+1] AS comp2
      MATCH (comp1)-[:LAST_DFS_NODE_IN_COMP]->(last)
      MERGE (last)-[:DFS_NEXT]->(comp2)
    }
    // Point all old heads to the new event
    CALL (e, sub_comps) {
      UNWIND sub_comps AS comp
      MERGE (comp)-[:COMPONENT_PARENT]->(e)
    }
    // Prepend new event and set new tail marker
    WITH event, e, sub_comps,
         sub_comps[0] AS first_comp,
         sub_comps[-1] AS last_comp
    MATCH (last_comp)-[:LAST_DFS_NODE_IN_COMP]->(last)
    MERGE (e)-[:DFS_NEXT]->(first_comp)
    MERGE (e)-[:LAST_DFS_NODE_IN_COMP]->(last)
    RETURN e
  }
}
RETURN e.event_id AS created_event
```

This single query handles all three cases atomically:
- **No match:** Creates a self-referencing single-event component.
- **Single match:** The old head gets `COMPONENT_PARENT` to the new event; the new event gets `DFS_NEXT` to the old head and inherits its `LAST_DFS_NODE_IN_COMP`.
- **Multi-match (merge):** All old heads get `COMPONENT_PARENT` to the new event; their DFS chains are concatenated; the new event is prepended with `DFS_NEXT` to the first sub-component and `LAST_DFS_NODE_IN_COMP` to the tail of the last.

> **Important:** Online ingestion must be single-threaded to maintain the forest invariant (COMPONENT_PARENT out-degree ≤ 1). The batch pipeline uses `ORDER BY rand()` and concurrent transactions because it processes independent components in parallel — but any given component is processed sequentially.


## Feature Engineering

With the component structure in place, features can be computed for each event using its point-in-time component. These features are stored directly on the Event node for downstream consumption.

### Example Features

**Component size** — the number of events in the component at that moment. A rapidly growing component may signal coordinated activity.

**Component diameter** — the longest shortest path in the bipartite Event–Thing subgraph induced by the component, normalized for bipartite hop distance. Measures how "spread out" the network of shared identifiers is.

**Component velocity** — events per second across the component's time span. A high velocity (many events in a short period) may indicate automated or coordinated behavior.

### Feature Computation Query

```cypher
CYPHER 25
MATCH (source:Event)
CALL (source) {

    // Retrieve point-in-time component
    WITH source, COLLECT {
        MATCH (source)(()-[:DFS_NEXT]->(evs))*(last)
              <-[:LAST_DFS_NODE_IN_COMP]-(source)
        UNWIND [source] + evs AS ev
        RETURN ev
    } AS cc_elements
    RETURN source, cc_elements, toString(source.event_id) AS graph_name
    NEXT

    // Compute features using GDS on the component subgraph
    CALL (source, cc_elements, graph_name) {
        UNWIND cc_elements AS element
        MATCH (element)
        OPTIONAL MATCH (element)-[:WITH]->(thing)
        WITH gds.graph.project(
            graph_name,
            element,
            thing,
            {},
            {undirectedRelationshipTypes: ['*']}
        ) AS graph
        CALL (cc_elements, graph_name, graph) {
            WHEN graph IS NOT NULL AND graph.relationshipCount > 0 THEN {
                UNWIND cc_elements AS element
                CALL gds.allShortestPaths.delta.stream(graph_name, {
                    sourceNode: element
                })
                YIELD path
                WITH length(path) AS length
                RETURN max(length) / 2 AS diameter
            }
            ELSE {
                RETURN INFINITY AS diameter
            }
        }
        CALL gds.graph.drop(graph_name, false)
        YIELD graphName
        WITH source, size(cc_elements) AS nb_elements, diameter
        SET source.component_size = nb_elements,
            source.component_diameter = diameter
        // Compute velocity
        WITH source, cc_elements
        UNWIND cc_elements AS element
        WITH source, element ORDER BY element.timestamp ASC
        WITH source, collect(element) AS ordered_elements
        WITH
          source,
          duration.inSeconds(
            ordered_elements[0].timestamp,
            ordered_elements[-1].timestamp
          ).seconds AS time_span,
          size(ordered_elements) AS nb_events
        SET source.component_velocity =
          CASE WHEN time_span > 0
          THEN toFloat(nb_events) / time_span
          ELSE 0.0
          END
    }
} IN 6 CONCURRENT TRANSACTIONS OF 100 ROWS
```

This query projects the bipartite subgraph for each component into GDS on the fly, computes the diameter via delta-stepping shortest paths, then derives velocity from the time span. Every feature is temporally accurate — computed against the component as it existed when that event occurred.


## Summary

The Point-in-Time WCC Model is a graph data model pattern for Neo4j that solves the problem of temporal leakage in WCC-based feature engineering. It stores the full evolution of component membership directly in the graph through three overlay relationships: `COMPONENT_PARENT` (temporal versioning), `DFS_NEXT` (pre-materialized traversal), and `LAST_DFS_NODE_IN_COMP` (bounded chain walks).

The model supports both batch construction (via GDS) for training pipelines and single-event online ingestion (via pure Cypher) for real-time inference. Component retrieval at any point in time is a linked-list walk with sub-millisecond latency — no WCC recomputation required.

The Event–Thing bipartite foundation makes the pattern domain-agnostic. Any use case that involves timestamped events sharing identifiers — fraud detection, identity resolution, network analysis, supply chain monitoring — can plug into this model with minimal adaptation.
